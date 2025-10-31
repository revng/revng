//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/IR/Use.h"
#include "llvm/Pass.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ABI/ModelHelpers.h"
#include "revng/InitModelTypes/InitModelTypes.h"
#include "revng/Model/Binary.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Support/Assert.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TypeNames/LLVMTypeNames.h"
#include "revng/TypeNames/ModelCBuilder.h"

static Logger Log{ "implicit-model-cast" };

using ValueTypeMap = std::map<const llvm::Value *, const model::UpcastableType>;
using ModelPromotedTypesMap = std::map<const llvm::Instruction *, ValueTypeMap>;

struct ImplicitModelCastPass : public llvm::FunctionPass {
public:
  static char ID;

  ImplicitModelCastPass() : FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LoadModelWrapperPass>();
  }

  bool process(llvm::Function &F, const model::Binary &Model);

private:
  bool collectTypeInfoForTypePromotion(llvm::Instruction *I,
                                       const model::Binary &Model);
  llvm::SmallSet<llvm::Use *, 4>
  getOperandsToPromote(llvm::Instruction *I,
                       const model::Binary &Model,
                       llvm::ModuleSlotTracker &MST);

  bool collectTypeInfoForPromotionForSingleValue(const llvm::Value *Value,
                                                 const model::Type &OperandType,
                                                 llvm::Instruction *I,
                                                 const model::Binary &Model);

private:
  ValueTypeMap TypeMap;
  ModelPromotedTypesMap PromotedTypes;
};

static void setImplicitModelCast(llvm::Value *ModelCastCallValue) {
  revng_assert(isCallToTagged(ModelCastCallValue, FunctionTags::ModelCast));

  llvm::CallInst *ModelCastCall = cast<llvm::CallInst>(ModelCastCallValue);
  auto IntType = llvm::IntegerType::getInt1Ty(ModelCastCall->getContext());
  auto IsImplicitConstant = llvm::ConstantInt::getSigned(IntType,
                                                         /*IsImplicit*/ true);
  ModelCastCall->setArgOperand(2, IsImplicitConstant);
}

// Return original or promoted type for a model type. Scalar type needs to be at
// least as int (32 bits), so if needed, we return a "promoted" version of the
// type. This is called "Integer Promotion".
static model::UpcastableType
getOriginalOrPromotedType(const model::Type &OperandType,
                          const model::Binary &Model) {

  const auto &Unwrapped = *OperandType.skipConstAndTypedefs();
  if (not Unwrapped.isObject() or not Unwrapped.isScalar())
    return model::UpcastableType::empty();

  if (OperandType.isPointer()) {
    return OperandType;

  } else if (const model::EnumDefinition *Enum = Unwrapped.getEnum()) {
    // Enum's underlying types will be processed later, when checking if a type
    // can be implicitly casted into another. Just make sure that enum's
    // underlying type is at least 4 bytes wide.
    if (Enum->size() >= 4u)
      return OperandType;
    else
      return model::UpcastableType::empty();

  } else if (const model::PrimitiveType *Primitive = Unwrapped.getPrimitive()) {
    if (Primitive->isFloatPrimitive())
      return model::UpcastableType::empty();

    model::UpcastableType Copy = *Primitive;
    if (Copy->toPrimitive().Size() < 4)
      Copy->toPrimitive().Size() = 4;
    return Copy;

  } else {
    revng_abort("Unsupported scalar type");
  }
}

// Check if we want to apply the Integer Promotion.
static bool shouldApplyIntegerPromotion(const llvm::Instruction *I) {
  if (isCallToTagged(I, FunctionTags::ModelCast))
    return true;

  if (auto *Bin = dyn_cast<llvm::BinaryOperator>(I)) {
    auto Opcode = Bin->getOpcode();
    switch (Opcode) {
    case llvm::Instruction::AShr:
    case llvm::Instruction::Shl:
    case llvm::Instruction::LShr:
    case llvm::Instruction::And:
    case llvm::Instruction::Or:
    case llvm::Instruction::Xor:
    case llvm::Instruction::Sub:
    case llvm::Instruction::Add:
    case llvm::Instruction::UDiv:
    case llvm::Instruction::URem:
    case llvm::Instruction::SDiv:
    case llvm::Instruction::SRem:
    case llvm::Instruction::Mul: {
      return true;
    }
    default:
      return false;
    }
  } else if (isa<llvm::SelectInst>(I) or isa<llvm::ICmpInst>(I)
             or isa<llvm::ReturnInst>(I)) {
    return true;
  }

  if (isCallToTagged(I, FunctionTags::Assign)
      or isCallToTagged(I, FunctionTags::UnaryMinus)
      or isCallToTagged(I, FunctionTags::BooleanNot)
      or isCallToTagged(I, FunctionTags::BinaryNot))
    return true;

  if (auto Call = dyn_cast<llvm::CallInst>(I)) {
    for (unsigned I = 0; I < Call->arg_size(); ++I) {
      auto Arg = Call->getArgOperand(I);
      if (isCallToTagged(Arg, FunctionTags::ModelCast))
        return true;
    }
  }

  // TODO: Make this Pass more robust by supporting more llvm::Instructions.

  return false;
}

// Check if a type can be casted into Target type, without losing precision.
static bool isImplicitCast(const model::Type &From,
                           const model::Type &To,
                           const llvm::Value *V) {
  // If types are the same, the cast is meaningless anyway.
  if (From == To)
    return true;

  // Force an explicit cast if it is discarding constness information.
  bool FromConst = From.isConst();
  bool ToConst = To.isConst();
  if (FromConst != ToConst and FromConst)
    return false;

  // If types only differ in typedefs or constness, make the cast implicit.
  const model::UpcastableType PostConstFrom = model::getNonConst(From);
  const model::UpcastableType PostConstTo = model::getNonConst(To);
  const model::Type *UnwrappedFrom = PostConstFrom->skipTypedefs();
  const model::Type *UnwrappedTo = PostConstTo->skipTypedefs();
  if (*UnwrappedFrom == *UnwrappedTo)
    return true;

  // Always preserve casts involving non-scalars.
  if (not UnwrappedFrom->isScalar() or not UnwrappedTo->isScalar())
    return false;

  if (const model::PointerType *FromPointer = UnwrappedFrom->getPointer()) {
    if (const model::PointerType *ToPointer = UnwrappedTo->getPointer()) {
      // Force an explicit cast if it is discarding constness information.
      bool FromConst = FromPointer->PointeeType()->isConst();
      bool ToConst = ToPointer->PointeeType()->isConst();
      if (FromConst != ToConst and FromConst)
        return false;

      // If pointee types only differ in typedefs or constness,
      // make the cast implicit.
      const auto NonConstFrom = model::getNonConst(*FromPointer->PointeeType());
      const auto NonConstTo = model::getNonConst(*ToPointer->PointeeType());
      const model::Type *UnwrappedFrom = NonConstFrom->skipTypedefs();
      const model::Type *UnwrappedTo = NonConstTo->skipTypedefs();
      if (*UnwrappedFrom == *UnwrappedTo)
        return true;

      // Always allow casts to void pointers.
      if (UnwrappedTo->isVoidPrimitive())
        return true;

      // Forbid all other pointer casts.
      // TODO: are there any others we want to allow?
      return false;
    }
  }

  // Explicitly forbid casts between pointers and non-pointers.
  if (UnwrappedFrom->isPointer() or UnwrappedTo->isPointer())
    return false;

  // Unwrap enums.
  if (const auto *Enum = UnwrappedFrom->getEnum())
    UnwrappedFrom = &Enum->underlyingType();
  if (const auto *Enum = UnwrappedTo->getEnum())
    UnwrappedTo = &Enum->underlyingType();

  // If we get to this point, both of those are always primitie.
  const model::PrimitiveType &FromPrimitive = UnwrappedFrom->toPrimitive();
  const model::PrimitiveType &ToPrimitive = UnwrappedTo->toPrimitive();

  auto ShouldTreatAsUnsigned = [](const model::PrimitiveType &Primitive) {
    return Primitive.PrimitiveKind() == model::PrimitiveKind::PointerOrNumber
           or Primitive.PrimitiveKind() == model::PrimitiveKind::Generic
           or Primitive.PrimitiveKind() == model::PrimitiveKind::Unsigned
           or Primitive.PrimitiveKind() == model::PrimitiveKind::Number;
  };

  if (ShouldTreatAsUnsigned(FromPrimitive)) {
    // Handle Unsigned to Unsigned.
    if (ShouldTreatAsUnsigned(ToPrimitive))
      return FromPrimitive.Size() <= ToPrimitive.Size();

    // Handle Unsigned to a Signed with larger size.
    if (ToPrimitive.PrimitiveKind() == model::PrimitiveKind::Signed)
      return FromPrimitive.Size() < ToPrimitive.Size();

  } else if (FromPrimitive.PrimitiveKind() == model::PrimitiveKind::Signed) {
    // Handle Signed to Signed.
    if (ToPrimitive.PrimitiveKind() == model::PrimitiveKind::Signed)
      return FromPrimitive.Size() < ToPrimitive.Size();
  }

  return false;
}

// This is to avoid -Wshift-count-overflow in C.
static bool isShiftCountGreaterThanExpectedType(const model::Type &Type,
                                                const llvm::Value *V) {
  if (auto *ModelCast = getCallToTagged(V, FunctionTags::ModelCast)) {
    llvm::Value *V = ModelCast->getArgOperand(1);
    if (auto *ShiftCount = llvm::dyn_cast<llvm::ConstantInt>(V)) {
      revng_assert(Type.size().has_value());
      if (ShiftCount->getZExtValue() >= *Type.size())
        return true;
    }
  }

  return false;
}

static bool isShiftLikeInstruction(llvm::Instruction *I) {
  if (I->getOpcode() == llvm::Instruction::Shl
      || I->getOpcode() == llvm::Instruction::AShr
      || I->getOpcode() == llvm::Instruction::LShr)
    return true;
  return false;
}

// Returns set of operand's indices that do not need a cast.
llvm::SmallSet<llvm::Use *, 4>
ImplicitModelCastPass::getOperandsToPromote(llvm::Instruction *I,
                                            const model::Binary &Model,
                                            llvm::ModuleSlotTracker &MST) {
  llvm::SmallSet<llvm::Use *, 4> Result;
  if (not shouldApplyIntegerPromotion(I)) {
    revng_log(Log,
              "Shouldn't be applying integer promotion for "
                << dumpToString(*I, MST));
    return Result;
  }

  auto GetPlainTypeName = [](const model::Type &Type) {
    ptml::ModelCBuilder B(llvm::nulls(), {}, /* TaglessMode = */ true);
    return B.getTypeName(Type);
  };

  for (unsigned Index = 0; Index < I->getNumOperands(); ++Index) {
    llvm::Use &Op = I->getOperandUse(Index);
    if (not isCallToTagged(Op.get(), FunctionTags::ModelCast))
      continue;

    if (not PromotedTypes.contains(I)) {
      revng_log(Log, "No promoted types for: " << dumpToString(*I, MST));
      return Result;
    }

    auto CallToModelCast = cast<llvm::CallInst>(Op.get());
    auto *CastedValue = CallToModelCast->getArgOperand(1);
    // Expected Type for the casted operand is the type of the cast, since the
    // MakeModelCast already made the cast.
    const model::Type &ExpectedType = *TypeMap.at(CallToModelCast);

    // Check if shift count < width of type.
    if (isShiftLikeInstruction(I) and Op.getOperandNo() == 0
        and isShiftCountGreaterThanExpectedType(ExpectedType,
                                                I->getOperandUse(1).get()))
      continue;

    auto PromotedTypesForInstruction = PromotedTypes[I];
    if (not PromotedTypesForInstruction.contains(CastedValue))
      continue;

    auto PromotedTypeForCastedValue = PromotedTypesForInstruction[CastedValue];
    const model::Type &CastedValueType = *TypeMap.at(CastedValue);
    // If type of the value being casted or integer promoted type are implicit
    // casts, we can avoid the cast itself.
    bool IsImplicit = isImplicitCast(*PromotedTypeForCastedValue,
                                     ExpectedType,
                                     CastedValue)
                      || isImplicitCast(CastedValueType,
                                        ExpectedType,
                                        CastedValue);

    if (not IsImplicit) {
      revng_log(Log,
                " `" << GetPlainTypeName(*PromotedTypeForCastedValue) << "` (`"
                     << GetPlainTypeName(CastedValueType)
                     << "`) CANNOT be implicitly cast to `"
                     << GetPlainTypeName(ExpectedType) << "`.");
      continue;
    } else {
      revng_log(Log,
                " '`" << GetPlainTypeName(*PromotedTypeForCastedValue) << "` (`"
                      << GetPlainTypeName(CastedValueType)
                      << "`) can be implicitly cast to `"
                      << GetPlainTypeName(ExpectedType) << "`.");
    }

    Result.insert(&Op);
  }

  return Result;
}

static bool isCandidate(llvm::Instruction &I) {
  return std::any_of(I.op_begin(), I.op_end(), [](llvm::Value *V) {
    return isCallToTagged(V, FunctionTags::ModelCast);
  });
}

using IMCP = ImplicitModelCastPass;
// Collect type information for a single operand from an LLVM instruction.
bool IMCP::collectTypeInfoForPromotionForSingleValue(const llvm::Value *Value,
                                                     const model::Type
                                                       &OperandType,
                                                     llvm::Instruction *I,
                                                     const model::Binary
                                                       &Model) {
  auto PromotedType = getOriginalOrPromotedType(OperandType, Model);
  if (PromotedType) {
    // This will be used in the second stage of the
    // reducing-cast-algorithm.
    auto &&[It, Success] = PromotedTypes[I].try_emplace(Value, PromotedType);
    if (not Success) {
      revng_assert(not It->second.isEmpty());
      revng_assert(not PromotedType.isEmpty());
      revng_assert(*It->second == *PromotedType);
    }

    return true;
  }

  return false;
}

// Collect type information for the operands from an LLVM instructions. It will
// be used for the type promotion later.
bool IMCP::collectTypeInfoForTypePromotion(llvm::Instruction *I,
                                           const model::Binary &Model) {
  using namespace model;
  using namespace abi::FunctionType;

  bool Result = false;
  auto CheckTypeFor = [this, &Model, &Result, &I](const llvm::Use &Op) {
    const model::Type *OperandType = nullptr;
    model::UpcastableType ExpectedType = model::UpcastableType::empty();
    llvm::Value *ValueToPromoteTypeFor = nullptr;
    if (isCallToTagged(Op.get(), FunctionTags::ModelCast)) {
      // If it is a ModelCast, get the type for the value being casted.
      // The expected type is the type of the model-cast, since it was
      // already "casted" by the MakeModelCast Pass.
      llvm::CallInst *CallToModelCast = cast<llvm::CallInst>(Op.get());
      llvm::Value *CastedValue = CallToModelCast->getArgOperand(1);
      OperandType = TypeMap.at(CastedValue).get();
      ValueToPromoteTypeFor = CastedValue;
      ExpectedType = TypeMap.at(Op.get());
    } else {
      // If it is not a ModelCast, promote the type for the llvm::Value itself.
      OperandType = TypeMap.at(Op.get()).get();
      ValueToPromoteTypeFor = Op.get();
      auto ModelTypes = getExpectedModelType(&Op, Model);
      if (ModelTypes.size() != 1)
        return;
      ExpectedType = std::move(ModelTypes.back());
    }

    revng_assert(OperandType != nullptr);
    revng_assert(OperandType->verify(true));
    revng_assert(not ExpectedType.isEmpty());
    revng_assert(ExpectedType->verify(true));
    if (OperandType->isScalar() and ExpectedType->isScalar()) {
      // Integer promotion - stage 1: Try to see if the operand can be
      // promoted implicitly into the expected type, so we can avoid
      // casts later (see stage 2).
      if (collectTypeInfoForPromotionForSingleValue(ValueToPromoteTypeFor,
                                                    *OperandType,
                                                    I,
                                                    Model)) {
        Result = true;
      }
    }
  };

  if (auto *Call = dyn_cast<llvm::CallInst>(I)) {
    auto *Callee = getCalledFunction(Call);
    if (isCallToIsolatedFunction(Call)) {
      if (not Callee)
        CheckTypeFor(Call->getCalledOperandUse());

      for (llvm::Use &Op : Call->args())
        CheckTypeFor(Op);
    } else if (FunctionTags::ModelGEP.isTagOf(Callee)
               or FunctionTags::ModelGEPRef.isTagOf(Callee)) {
      // Check the type of the base operand
      CheckTypeFor(Call->getArgOperandUse(1));

      // If there are other arguments past the first two that are not constant
      // indices, it means that they are indices into an array. For those, we
      // have to make sure they are integers, possibly injecting casts.
      if (Call->arg_size() > 2) {
        for (llvm::Use &Argument : llvm::drop_begin(Call->args(), 2)) {
          CheckTypeFor(Argument);
        }
      }

    } else if (FunctionTags::AddressOf.isTagOf(Callee)) {
      // Check the type of the base operand
      CheckTypeFor(Call->getArgOperandUse(1));

    } else if (FunctionTags::BinaryNot.isTagOf(Callee)) {
      CheckTypeFor(Call->getArgOperandUse(0));
    } else if (FunctionTags::StructInitializer.isTagOf(Callee)) {
      for (llvm::Use &Op : Call->args())
        CheckTypeFor(Op);
    } else if (FunctionTags::ModelCast.isTagOf(Callee)) {
      CheckTypeFor(I->getOperandUse(1));
    } else if (isCallToTagged(I, FunctionTags::Assign)
               or isCallToTagged(I, FunctionTags::UnaryMinus)
               or isCallToTagged(I, FunctionTags::BooleanNot)
               or isCallToTagged(I, FunctionTags::BinaryNot)) {
      CheckTypeFor(I->getOperandUse(0));
    }
  } else if (auto *Ret = dyn_cast<llvm::ReturnInst>(I)) {
    if (Ret->getNumOperands() > 0)
      CheckTypeFor(Ret->getOperandUse(0));
  } else if (isa<llvm::BinaryOperator>(I) or isa<llvm::ICmpInst>(I)
             or isa<llvm::SelectInst>(I)) {
    for (unsigned Index = 0; Index < I->getNumOperands(); ++Index) {
      CheckTypeFor(I->getOperandUse(Index));
    }
  } else if (auto *Switch = dyn_cast<llvm::SwitchInst>(I)) {
    CheckTypeFor(Switch->getOperandUse(0));
  }

  return Result;
}

bool ImplicitModelCastPass::process(llvm::Function &F,
                                    const model::Binary &Model) {
  bool Changed = false;

  llvm::ModuleSlotTracker MST(F.getParent(),
                              /* ShouldInitializeAllMetadata = */ false);
  if (Log.isEnabled())
    MST.incorporateFunction(F);

  for (llvm::BasicBlock &BB : F) {
    for (llvm::Instruction &I : BB) {
      // Candidates for the cast reduction are instructions that contain
      // model-cast op.
      if (not isCandidate(I) or not shouldApplyIntegerPromotion(&I))
        continue;

      bool HasTypes = collectTypeInfoForTypePromotion(&I, Model);
      if (not HasTypes) {
        revng_log(Log,
                  "No types collected to promote for " << dumpToString(I, MST));
        continue;
      }

      // Integer promotion - stage 2: Try to avoid casts that are not needed in
      // C.
      llvm::SmallSet<llvm::Use *, 4>
        OperandsToPromoteTypeFor = getOperandsToPromote(&I, Model, MST);
      if (not OperandsToPromoteTypeFor.size()) {
        revng_log(Log,
                  "No operands detected for type promotion in "
                    << dumpToString(I, MST));
        continue;
      }

      revng_log(Log, "Marking implicit casts in " << dumpToString(I, MST));
      // Here we try to mark some casts as implicit by changing the LLVM IR.
      for (unsigned Index = 0; Index < I.getNumOperands(); ++Index) {
        llvm::Use &OperandUse = I.getOperandUse(Index);
        if (OperandsToPromoteTypeFor.contains(&OperandUse)) {
          revng_log(Log,
                    " Found implicit cast: " << dumpToString(*OperandUse.get(),
                                                             MST));
          setImplicitModelCast(OperandUse.get());
          Changed = true;
        }
      }
    }
  }

  return Changed;
}

bool ImplicitModelCastPass::runOnFunction(llvm::Function &F) {
  bool Changed = false;

  auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();
  const TupleTree<model::Binary> &Model = ModelWrapper.getReadOnlyModel();

  auto ModelFunction = llvmToModelFunction(*Model, F);
  revng_assert(ModelFunction != nullptr);

  TypeMap = initModelTypes(F, ModelFunction, *Model, false);

  Changed = process(F, *Model);

  return Changed;
}

char ImplicitModelCastPass::ID = 0;

using P = ImplicitModelCastPass;
static llvm::RegisterPass<P> X("implicit-model-cast", "ImplicitCasts");
