//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Use.h"
#include "llvm/Pass.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/YAMLTraits.h"

#include "revng-c/InitModelTypes/InitModelTypes.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/IRHelpers.h"
#include "revng-c/Support/ModelHelpers.h"
#include "revng-c/TypeNames/LLVMTypeNames.h"

static Logger<> Log{ "implicit-model-cast" };

using namespace llvm;
using model::QualifiedType;

using ValueTypeMap = std::map<const llvm::Value *, const model::QualifiedType>;
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

  bool process(Function &F, const model::Binary &Model);

private:
  bool collectTypeInfoForTypePromotion(Instruction *I,
                                       const model::Binary &Model);
  SmallSet<llvm::Use *, 4> getOperandsToPromote(Instruction *I,
                                                const model::Binary &Model);

  bool collectTypeInfoForPromotionForSingleValue(const Value *Value,
                                                 const model::QualifiedType
                                                   &OperandType,
                                                 Instruction *I,
                                                 const model::Binary &Model);

private:
  ValueTypeMap TypeMap;
  ModelPromotedTypesMap PromotedTypes;
};

static void setImplicitModelCast(Value *ModelCastCallValue) {
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
static std::optional<model::QualifiedType>
getOriginalOrPromotedType(const model::QualifiedType &OperandType,
                          const model::Binary &Model) {
  if (OperandType.isPointer())
    return OperandType;

  auto UnwrappedType = peelConstAndTypedefs(OperandType);
  if (not UnwrappedType.isScalar())
    return std::nullopt;

  revng_assert(UnwrappedType.Qualifiers().empty());

  // Enum's underlying types will be processed later, when checking if a type
  // can be implicitly casted into another. Just make sure that enum's
  // underlying type is at least 4 bytes wide.
  if (llvm::isa<model::EnumType>(UnwrappedType.UnqualifiedType().getConst())) {
    auto SizeOfTheType = OperandType.size();
    if (SizeOfTheType and *SizeOfTheType >= 4u)
      return OperandType;
    else
      return std::nullopt;
  }

  auto AsPrimitive = cast<model::PrimitiveType>(UnwrappedType.UnqualifiedType()
                                                  .getConst());
  auto TypeKind = AsPrimitive->PrimitiveKind();
  if (TypeKind == model::PrimitiveTypeKind::Void
      or TypeKind == model::PrimitiveTypeKind::Float)
    return std::nullopt;

  unsigned ByteSize = AsPrimitive->Size();
  return model::QualifiedType{
    Model.getPrimitiveType(TypeKind, std::max(4u, ByteSize)), {}
  };
}

// Check if we want to apply the Integer Promotion.
static bool shouldApplyIntegerPromotion(const Instruction *I) {
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

// Check if QT can be casted into Target type, without losing precision.
static bool isImplicitCast(const model::QualifiedType &QT,
                           const model::QualifiedType &Target,
                           const llvm::Value *V) {
  using model::PrimitiveTypeKind::Generic;
  using model::PrimitiveTypeKind::Number;
  using model::PrimitiveTypeKind::PointerOrNumber;
  using model::PrimitiveTypeKind::Signed;
  using model::PrimitiveTypeKind::Unsigned;
  using model::PrimitiveTypeKind::Void;

  if (QT == Target)
    return true;

  // This will automatically handle typedefs as well.
  auto UnwrappedQTType = peelConstAndTypedefs(QT);
  auto UnwrappedTargetType = peelConstAndTypedefs(Target);

  if (not UnwrappedQTType.isScalar() or not UnwrappedTargetType.isScalar())
    return false;

  if (UnwrappedQTType == UnwrappedTargetType)
    return true;

  // A non-const pointer to void * is an implicit cast.
  if (UnwrappedQTType.isPointer() and UnwrappedTargetType.isPointer()) {
    const model::QualifiedType PointeeQT = UnwrappedQTType.stripPointer();
    const model::QualifiedType PointeeTarget = UnwrappedTargetType
                                                 .stripPointer();
    if (PointeeQT.isConst() or PointeeTarget.isConst())
      return false;

    if (UnwrappedTargetType.isVoid())
      return true;
    return false;
  }

  if (UnwrappedQTType.isPointer() or UnwrappedTargetType.isPointer())
    return UnwrappedQTType == UnwrappedTargetType;

  if (const auto *QTAsEnum = dyn_cast<model::EnumType>(UnwrappedQTType
                                                         .UnqualifiedType()
                                                         .getConst())) {
    UnwrappedQTType = QTAsEnum->UnderlyingType();
  }

  if (const auto *TargetAsEnum = dyn_cast<model::EnumType>(UnwrappedTargetType
                                                             .UnqualifiedType()
                                                             .getConst())) {
    UnwrappedTargetType = TargetAsEnum->UnderlyingType();
  }

  revng_assert(UnwrappedQTType.Qualifiers().empty());
  revng_assert(UnwrappedTargetType.Qualifiers().empty());

  auto TargetAsPrimitive = cast<model::PrimitiveType>(UnwrappedTargetType
                                                        .UnqualifiedType()
                                                        .getConst());
  auto QTAsPrimitive = cast<model::PrimitiveType>(UnwrappedQTType
                                                    .UnqualifiedType()
                                                    .getConst());
  switch (QTAsPrimitive->PrimitiveKind()) {
  // This will be identical in C, see `revng-primitive-types.h`.
  case Generic:
  case Unsigned:
  case Number:
  case PointerOrNumber: {
    auto TargetKind = TargetAsPrimitive->PrimitiveKind();

    // Handle Unsigned to Unsigned.
    if (TargetKind == Generic or TargetKind == Unsigned or TargetKind == Number
        or TargetKind == PointerOrNumber) {
      return QTAsPrimitive->Size() <= TargetAsPrimitive->Size();
    }

    // Handle Unsigned to a Signed with larger size.
    if (TargetKind == Signed)
      return QTAsPrimitive->Size() < TargetAsPrimitive->Size();

    break;
  }
  case Signed: {
    auto TargetKind = TargetAsPrimitive->PrimitiveKind();
    if (TargetKind == Signed)
      return QTAsPrimitive->Size() < TargetAsPrimitive->Size();
    break;
  }
  default:
    break;
  }

  return false;
}

// This is to avoid -Wshift-count-overflow in C.
static bool
isShiftCountGreaterThanExpectedType(const model::QualifiedType &Type,
                                    const llvm::Value *V) {
  if (auto *ModelCast = getCallToTagged(V, FunctionTags::ModelCast)) {
    llvm::Value *V = ModelCast->getArgOperand(1);
    if (llvm::isa<ConstantInt>(V)) {
      uint64_t ShiftCountValue = cast<ConstantInt>(V)->getZExtValue();
      auto TypeSize = Type.size();
      if (TypeSize and ShiftCountValue >= *TypeSize)
        return true;
    }
  }

  return false;
}

static bool isShiftLikeInstruction(Instruction *I) {
  if (I->getOpcode() == llvm::Instruction::Shl
      || I->getOpcode() == llvm::Instruction::AShr
      || I->getOpcode() == llvm::Instruction::LShr)
    return true;
  return false;
}

// Returns set of operand's indices that do not need a cast.
SmallSet<llvm::Use *, 4>
ImplicitModelCastPass::getOperandsToPromote(Instruction *I,
                                            const model::Binary &Model) {
  SmallSet<llvm::Use *, 4> Result;
  if (not shouldApplyIntegerPromotion(I)) {
    revng_log(Log,
              "Shouldn't be applying integer promotion for "
                << dumpToString(I));
    return Result;
  }

  for (unsigned Index = 0; Index < I->getNumOperands(); ++Index) {
    llvm::Use &Op = I->getOperandUse(Index);
    if (not isCallToTagged(Op.get(), FunctionTags::ModelCast))
      continue;

    if (not PromotedTypes.contains(I)) {
      revng_log(Log, "No promoted types for: " << dumpToString(I));
      return Result;
    }

    auto CallToModelCast = cast<llvm::CallInst>(Op.get());
    auto *CastedValue = CallToModelCast->getArgOperand(1);
    // Expected Type for the casted operand is the type of the cast, since the
    // MakeModelCast already made the cast.
    QualifiedType ExpectedType = TypeMap.at(CallToModelCast);

    // Check if shift count < width of type.
    if (isShiftLikeInstruction(I) and Op.getOperandNo() == 0
        and isShiftCountGreaterThanExpectedType(ExpectedType,
                                                I->getOperandUse(1).get()))
      continue;

    auto PromotedTypesForInstruction = PromotedTypes[I];
    if (not PromotedTypesForInstruction.contains(CastedValue))
      continue;

    auto PromotedTypeForCastedValue = PromotedTypesForInstruction[CastedValue];
    QualifiedType CastedValueType = TypeMap.at(CastedValue);
    // If type of the value being casted or integer promoted type are implicit
    // casts, we can avoid the cast itself.
    if (not isImplicitCast(PromotedTypeForCastedValue,
                           ExpectedType,
                           CastedValue)
        and not isImplicitCast(CastedValueType, ExpectedType, CastedValue)) {
      continue;
    }

    Result.insert(&Op);
  }

  return Result;
}

static bool isCandidate(Instruction &I) {
  return std::any_of(I.op_begin(), I.op_end(), [](Value *V) {
    return isCallToTagged(V, FunctionTags::ModelCast);
  });
}

using IMCP = ImplicitModelCastPass;
// Collect type information for a single operand from an LLVM instruction.
bool IMCP::collectTypeInfoForPromotionForSingleValue(const Value *Value,
                                                     const model::QualifiedType
                                                       &OperandType,
                                                     Instruction *I,
                                                     const model::Binary
                                                       &Model) {
  auto PromotedType = getOriginalOrPromotedType(OperandType, Model);
  if (PromotedType) {
    // This will be used in the second stage of the
    // reducing-cast-algorithm.
    PromotedTypes[I].insert({ Value, *PromotedType });
    return true;
  }

  return false;
}

// Collect type information for the operands from an LLVM instructions. It will
// be used for the type promotion later.
bool IMCP::collectTypeInfoForTypePromotion(Instruction *I,
                                           const model::Binary &Model) {
  using namespace model;
  using namespace abi::FunctionType;

  bool Result = false;
  auto CheckTypeFor = [this, &Model, &Result, &I](const llvm::Use &Op) {
    std::optional<QualifiedType> OperandType;
    std::optional<QualifiedType> ExpectedType;
    Value *ValueToPromoteTypeFor = nullptr;
    if (isCallToTagged(Op.get(), FunctionTags::ModelCast)) {
      // If it is a ModelCast, get the type for the value being casted.
      // The expected type is the type of the model-cast, since it was
      // already "casted" by the MakeModelCast Pass.
      llvm::CallInst *CallToModelCast = cast<llvm::CallInst>(Op.get());
      llvm::Value *CastedValue = CallToModelCast->getArgOperand(1);
      OperandType = TypeMap.at(CastedValue);
      ValueToPromoteTypeFor = CastedValue;
      ExpectedType = TypeMap.at(Op.get());
    } else {
      // If it is not a ModelCast, promote the type for the llvm::Value itself.
      OperandType = TypeMap.at(Op.get());
      ValueToPromoteTypeFor = Op.get();
      auto ModelTypes = getExpectedModelType(&Op, Model);
      if (ModelTypes.size() != 1)
        return;
      ExpectedType = ModelTypes.back();
    }

    revng_assert(ExpectedType->UnqualifiedType().isValid());
    revng_assert(OperandType->UnqualifiedType().isValid());
    if (ExpectedType->isScalar() and OperandType->isScalar()) {
      // Integer promotion - stage 1: Try to see if the operand can be
      // promoted implicitly into the expected type, so we can avoid
      // casts later (see stage 2).
      Result |= collectTypeInfoForPromotionForSingleValue(ValueToPromoteTypeFor,
                                                          *OperandType,
                                                          I,
                                                          Model);
    }
  };

  if (auto *Call = dyn_cast<CallInst>(I)) {
    auto *Callee = Call->getCalledFunction();
    if (isCallToIsolatedFunction(Call)) {
      if (not Callee)
        CheckTypeFor(Call->getCalledOperandUse());

      for (llvm::Use &Op : Call->args())
        CheckTypeFor(Op);
    } else if (FunctionTags::ModelGEP.isTagOf(Callee)
               or FunctionTags::ModelGEPRef.isTagOf(Callee)
               or FunctionTags::AddressOf.isTagOf(Callee)) {
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
  } else if (auto *Ret = dyn_cast<ReturnInst>(I)) {
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

bool ImplicitModelCastPass::process(Function &F, const model::Binary &Model) {
  bool Changed = false;

  for (llvm::BasicBlock &BB : F) {
    for (llvm::Instruction &I : BB) {
      // Candidates for the cast reduction are instructions that contain
      // model-cast op.
      if (not isCandidate(I) or not shouldApplyIntegerPromotion(&I))
        continue;

      bool HasTypes = collectTypeInfoForTypePromotion(&I, Model);
      if (not HasTypes) {
        revng_log(Log, "No types collected to promote for " << dumpToString(I));
        continue;
      }

      // Integer promotion - stage 2: Try to avoid casts that are not needed in
      // C.
      SmallSet<llvm::Use *, 4>
        OperandsToPromoteTypeFor = getOperandsToPromote(&I, Model);
      if (not OperandsToPromoteTypeFor.size()) {
        revng_log(Log,
                  "No operands detected for type promotion in "
                    << dumpToString(I));
        continue;
      }

      revng_log(Log, "Marking implicit casts in " << dumpToString(I));
      // Here we try to mark some casts as implicit by changing the LLVM IR.
      for (unsigned Index = 0; Index < I.getNumOperands(); ++Index) {
        llvm::Use &OperandUse = I.getOperandUse(Index);
        if (OperandsToPromoteTypeFor.contains(&OperandUse)) {
          revng_log(Log,
                    " Found implicit cast: " << dumpToString(OperandUse.get()));
          setImplicitModelCast(OperandUse.get());
          Changed = true;
        }
      }
    }
  }

  return Changed;
}

bool ImplicitModelCastPass::runOnFunction(Function &F) {
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
static RegisterPass<P> X("implicit-model-cast", "ImplicitCasts", false, false);
