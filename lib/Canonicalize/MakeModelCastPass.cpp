//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Use.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ABI/ModelHelpers.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/InitModelTypes/InitModelTypes.h"
#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TypeNames/LLVMTypeNames.h"

static Logger<> Log{ "make-model-cast" };

using namespace llvm;

using TypePair = FunctionTags::TypePair;

using ModelTypesMap = std::map<const llvm::Value *,
                               const model::UpcastableType>;

struct CastToEmit {
  Use &OperandToCast;
  const model::UpcastableType TargetType;
};

struct MakeModelCastPass : public llvm::FunctionPass {
private:
  ModelTypesMap TypeMap;
  const model::Binary *Model;

public:
  static char ID;

  MakeModelCastPass() : FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LoadModelWrapperPass>();
  }

private:
  std::optional<CastToEmit> computeCast(Use &U) const;

  std::vector<CastToEmit> computeCasts(Instruction *I) const;

  void makeModelCast(const CastToEmit &ToEmit,
                     OpaqueFunctionsPool<TypePair> &Pool) const;
};

std::optional<CastToEmit> MakeModelCastPass::computeCast(Use &Operand) const {

  // Check if we have strong model information about this operand
  auto ModelTypes = getExpectedModelType(&Operand, *Model);

  // Aggregates that do not correspond to model structs (e.g. return types
  // of RawFunctionTypes that return more than one value) cannot be handled
  // with casts, since we don't have a model::TypeDefinition to cast them to.
  if (ModelTypes.size() == 1) {
    const model::UpcastableType &ExpectedType = ModelTypes.back();
    revng_assert(ExpectedType->verify());

    const model::Type &OperandType = *TypeMap.at(Operand.get());
    if (*ExpectedType->skipTypedefs() != *OperandType.skipTypedefs()) {
      revng_log(Log,
                "New CastToEmit on use of: "
                  << dumpToString(Operand.get())
                  << " in : " << dumpToString(Operand.getUser()));
      revng_log(Log,
                "Casting from : " << OperandType << " to : " << *ExpectedType);
      return CastToEmit{ .OperandToCast = Operand, .TargetType = ExpectedType };
    }
  }
  return std::nullopt;
}

std::vector<CastToEmit> MakeModelCastPass::computeCasts(Instruction *I) const {
  std::vector<CastToEmit> CastsToEmit;

  const auto PushBackMoving = [&CastsToEmit](std::optional<CastToEmit> &&C) {
    if (C.has_value())
      CastsToEmit.push_back(std::move(C.value()));
  };

  if (auto *Call = dyn_cast<CallInst>(I)) {
    // Lifted functions have their prototype on the model
    auto *Callee = getCalledFunction(Call);

    if (isCallToIsolatedFunction(Call)) {
      // For indirect calls, cast the callee to the right function type
      if (not Callee)
        PushBackMoving(computeCast(Call->getCalledOperandUse()));

      // For all calls, check if we need to cast the actual types to the formal
      // arguments types
      for (Use &Op : Call->args())
        PushBackMoving(computeCast(Op));

    } else if (FunctionTags::ModelGEP.isTagOf(Callee)
               or FunctionTags::ModelGEPRef.isTagOf(Callee)
               or FunctionTags::AddressOf.isTagOf(Callee)) {
      // Check the type of the base operand
      PushBackMoving(computeCast(Call->getArgOperandUse(1)));

    } else if (FunctionTags::BinaryNot.isTagOf(Callee)) {
      PushBackMoving(computeCast(Call->getArgOperandUse(0)));

    } else if (FunctionTags::StructInitializer.isTagOf(Callee)) {
      // StructInitializers are used to pack together a returned struct, so
      // we know the types of each element by looking at the Prototype
      for (llvm::Use &Op : Call->args())
        PushBackMoving(computeCast(Op));
    }

  } else if (auto *Ret = dyn_cast<ReturnInst>(I)) {
    // Check the formal return type
    if (Ret->getNumOperands() > 0)
      PushBackMoving(computeCast(Ret->getOperandUse(0)));

  } else if (isa<llvm::BinaryOperator>(I) or isa<llvm::ICmpInst>(I)
             or isa<llvm::SelectInst>(I)) {
    for (Use &Op : I->operands())
      PushBackMoving(computeCast(Op));

  } else if (auto *Switch = dyn_cast<llvm::SwitchInst>(I)) {
    PushBackMoving(computeCast(Switch->getOperandUse(0)));
  }

  return CastsToEmit;
}

static FunctionType *getModelCastType(TypePair Key) {
  LLVMContext &LLVMCtxt = Key.RetType->getContext();
  Type *StringPtrType = getStringPtrType(LLVMCtxt);
  IntegerType *Int1 = llvm::IntegerType::getInt1Ty(LLVMCtxt);
  return FunctionType::get(Key.RetType,
                           { StringPtrType, Key.ArgType, Int1 },
                           /* VarArg */ false);
}

static Function *
getModelCastFunction(TypePair Key,
                     OpaqueFunctionsPool<TypePair> &ModelCastPool) {
  FunctionType *ModelCastType = getModelCastType(Key);
  return ModelCastPool.get(Key, ModelCastType, "ModelCast");
}

// Create a call to explicit ModelCast. Later on, we run `ImplicitModelCastPass`
// to detect implicit casts.
static CallInst *createCallToModelCast(IRBuilder<> &Builder,
                                       TypePair Key,
                                       const model::UpcastableType &TargetType,
                                       Value *Operand,
                                       OpaqueFunctionsPool<TypePair> &Pool) {
  auto *ModelCastFunction = getModelCastFunction(Key, Pool);
  LLVMContext &Context = Builder.getContext();
  ConstantInt *IsImplicit = llvm::ConstantInt::getFalse(Context);
  CallInst
    *Call = Builder.CreateCall(ModelCastFunction,
                               { toLLVMString(TargetType,
                                              *ModelCastFunction->getParent()),
                                 Operand,
                                 IsImplicit });
  return Call;
}

void MakeModelCastPass::makeModelCast(const CastToEmit &ToEmit,
                                      OpaqueFunctionsPool<TypePair> &Pool)
  const {
  const auto &[OperandUse, TargetType] = ToEmit;

  Value *Operand = OperandUse.get();
  const model::Type &OperandModelType = *TypeMap.at(Operand);

  auto *I = cast<Instruction>(OperandUse.getUser());
  IRBuilder<> Builder(I);
  Type *OperandType = Operand->getType();
  CallInst *CallToModelCast = createCallToModelCast(Builder,
                                                    { OperandType,
                                                      OperandType },
                                                    TargetType,
                                                    Operand,
                                                    Pool);
  // If either the source type or the target type, on the model, are not scalar
  // types, we'll get a cast in C involving aggregates. This is guaranteed not
  // to be syntactically valid C code, but the current LLVM-based decompilation
  // pipeline has not way to strongly guarantee this never happens, so the only
  // thing we can do is to print a warning.
  // This problem will just go away with the clift-based backend.
  if (not TargetType->isScalar() or not OperandModelType.isScalar()) {
    std::string Warning;
    {
      llvm::raw_string_ostream OS{ Warning };

      OS << "WARNING: ModelCast involves non-scalar types. "
            "This may not compile in C.\n";

      OS << "OperandType: ";
      OperandModelType.dump(OS);
      OS << '\n';

      OS << "TargetType: ";
      TargetType->dump(OS);
      OS << '\n';

      OS.flush();
    }
    revng_log(Log, Warning);
  }
  OperandUse.set(CallToModelCast);

  revng_log(Log, "makeModelCast: " << dumpToString(CallToModelCast));
}

bool MakeModelCastPass::runOnFunction(Function &F) {
  bool Changed = false;

  Module *M = F.getParent();
  auto ModelCastPool = FunctionTags::ModelCast.getPool(*M);

  auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();
  Model = &*ModelWrapper.getReadOnlyModel();

  const model::Function *ModelFunction = llvmToModelFunction(*Model, F);
  revng_assert(ModelFunction != nullptr);

  revng_log(Log, "analyzing Function: " << F.getName());
  // First of all, remove all SExt, ZExt and Trunc, and replace them with
  // ModelCasts.
  {
    LoggerIndent Indent{ Log };
    revng_log(Log, "replacing sext/zext/trunc with ModelCast");

    LLVMContext &LLVMCtxt = F.getContext();
    IRBuilder<> Builder(LLVMCtxt);
    for (BasicBlock &BB : F) {
      for (Instruction &I : llvm::make_early_inc_range(BB)) {
        auto *SExt = dyn_cast<llvm::SExtInst>(&I);
        auto *ZExt = dyn_cast<llvm::ZExtInst>(&I);
        auto *Trunc = dyn_cast<llvm::TruncInst>(&I);
        if (not SExt and not ZExt and not Trunc)
          continue;

        LoggerIndent MoreIndent{ Log };
        revng_log(Log, "replacing: " << dumpToString(I));

        llvm::Value *CastedOperand = I.getOperand(0);

        // Build a FunctionType for the ModelCast function, and add it to the
        // pool
        auto *ResultTypeOnLLVM = cast<IntegerType>(I.getType());
        TypePair Key = TypePair{ .RetType = ResultTypeOnLLVM,
                                 .ArgType = CastedOperand->getType() };
        // Create the ModelCast call.
        Builder.SetInsertPoint(&I);

        // Compute the target type of the cast, depending on the cast we're
        // replacing.

        unsigned ResultBitWidth = ResultTypeOnLLVM->getBitWidth();
        revng_assert(std::has_single_bit(ResultBitWidth));
        revng_assert(ResultBitWidth == 1 or ResultBitWidth >= 8);
        unsigned ByteSize = (ResultBitWidth == 1) ? 1 : ResultBitWidth / 8;
        auto ResultModelType = SExt ?
                                 model::PrimitiveType::makeSigned(ByteSize) :
                               ZExt ?
                                 model::PrimitiveType::makeUnsigned(ByteSize) :
                                 model::PrimitiveType::makeNumber(ByteSize);

        CallInst *CallToModelCast = createCallToModelCast(Builder,
                                                          Key,
                                                          ResultModelType,
                                                          CastedOperand,
                                                          ModelCastPool);
        I.replaceAllUsesWith(CallToModelCast);
        I.eraseFromParent();

        revng_log(Log, "with: " << dumpToString(CallToModelCast));
      }
    }
  }

  TypeMap = initModelTypes(F, ModelFunction, *Model, false);

  for (BasicBlock &BB : F) {
    for (Instruction &I : llvm::make_early_inc_range(BB)) {
      LoggerIndent Indent{ Log };
      revng_log(Log, "computeCasts on operands of: " << dumpToString(I));
      LoggerIndent MoreIndent{ Log };

      std::vector<CastToEmit> ToEmit = computeCasts(&I);

      if (ToEmit.empty()) {
        revng_log(Log, "no casts!");
        continue;
      }

      Changed = true;

      for (const CastToEmit &C : ToEmit) {
        makeModelCast(C, ModelCastPool);
      }
    }
  }

  return Changed;
}

char MakeModelCastPass::ID = 0;

using Pass = MakeModelCastPass;
static RegisterPass<MakeModelCastPass> X("make-model-cast",
                                         "A pass that pulls out casts from "
                                         "some instructions that embed casts "
                                         "into their own dedicated calls.",
                                         false,
                                         false);
