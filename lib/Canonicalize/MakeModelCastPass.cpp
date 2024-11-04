//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Use.h"
#include "llvm/Pass.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/YAMLTraits.h"

#include "revng-c/InitModelTypes/InitModelTypes.h"
#include "revng-c/Support/ModelHelpers.h"
#include "revng-c/TypeNames/LLVMTypeNames.h"

using namespace llvm;

using TypePair = FunctionTags::TypePair;

using ModelTypesMap = std::map<const llvm::Value *,
                               const model::UpcastableType>;

struct SerializedType {
  Constant *StringType = nullptr;
  uint64_t OperandId;

  SerializedType(Constant *StringType, uint64_t OperandId = 0) :
    StringType(StringType), OperandId(OperandId) {}
};

struct MakeModelCastPass : public llvm::FunctionPass {
private:
  ModelTypesMap TypeMap;
  const model::Function *ModelFunction = nullptr;

public:
  static char ID;

  MakeModelCastPass() : FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LoadModelWrapperPass>();
  }

private:
  std::vector<SerializedType> serializeTypesForModelCast(Instruction *,
                                                         const model::Binary &);
  void createAndInjectModelCast(Instruction *,
                                const SerializedType &,
                                OpaqueFunctionsPool<TypePair> &);
};

using MMCP = MakeModelCastPass;

std::vector<SerializedType>
MMCP::serializeTypesForModelCast(Instruction *I, const model::Binary &Model) {
  using namespace model;
  using namespace abi::FunctionType;

  std::vector<SerializedType> Result;
  Module *M = I->getModule();

  auto SerializeTypeFor = [this, &Model, &Result, &M](const llvm::Use &Op) {
    // Check if we have strong model information about this operand
    auto ModelTypes = getExpectedModelType(&Op, Model);

    // Aggregates that do not correspond to model structs (e.g. return types
    // of RawFunctionTypes that return more than one value) cannot be handled
    // with casts, since we don't have a model::TypeDefinition to cast them to.
    if (ModelTypes.size() == 1) {
      const model::UpcastableType &ExpectedType = ModelTypes.back();
      revng_assert(ExpectedType->verify());

      const model::Type &OperandType = *TypeMap.at(Op.get());
      if (*ExpectedType->skipTypedefs() != *OperandType.skipTypedefs()) {
        revng_assert(ExpectedType->isScalar() and OperandType.isScalar());
        // Create a cast only if the expected type is different from the
        // actual type propagated until here
        auto Type = SerializedType(toLLVMString(ExpectedType, *M),
                                   Op.getOperandNo());
        Result.emplace_back(std::move(Type));
      }
    }
  };

  if (auto *Call = dyn_cast<CallInst>(I)) {
    // Lifted functions have their prototype on the model
    auto *Callee = getCalledFunction(Call);

    if (isCallToIsolatedFunction(Call)) {
      // For indirect calls, cast the callee to the right function type
      if (not Callee)
        SerializeTypeFor(Call->getCalledOperandUse());

      // For all calls, check the formal arguments types
      for (llvm::Use &Op : Call->args())
        SerializeTypeFor(Op);

    } else if (FunctionTags::ModelGEP.isTagOf(Callee)
               or FunctionTags::ModelGEPRef.isTagOf(Callee)
               or FunctionTags::AddressOf.isTagOf(Callee)) {
      // Check the type of the base operand
      SerializeTypeFor(Call->getArgOperandUse(1));
    } else if (FunctionTags::BinaryNot.isTagOf(Callee)) {
      SerializeTypeFor(Call->getArgOperandUse(0));
    } else if (FunctionTags::StructInitializer.isTagOf(Callee)) {
      // StructInitializers are used to pack together a returned struct, so
      // we know the types of each element by looking at the Prototype
      for (llvm::Use &Op : Call->args())
        SerializeTypeFor(Op);
    }
  } else if (auto *Ret = dyn_cast<ReturnInst>(I)) {
    // Check the formal return type
    if (Ret->getNumOperands() > 0)
      SerializeTypeFor(Ret->getOperandUse(0));

  } else if (isa<llvm::BinaryOperator>(I) or isa<llvm::ICmpInst>(I)
             or isa<llvm::SelectInst>(I)) {
    for (unsigned Index = 0; Index < I->getNumOperands(); ++Index)
      SerializeTypeFor(I->getOperandUse(Index));
  } else if (auto *Switch = dyn_cast<llvm::SwitchInst>(I)) {
    SerializeTypeFor(Switch->getOperandUse(0));
  }

  return Result;
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
static Value *createCallToModelCast(IRBuilder<> &Builder,
                                    TypePair Key,
                                    Constant *SerializedTypeAsString,
                                    Value *Operand,
                                    OpaqueFunctionsPool<TypePair> &Pool) {
  auto *ModelCastFunction = getModelCastFunction(Key, Pool);
  LLVMContext &Context = Builder.getContext();
  ConstantInt *IsImplicit = llvm::ConstantInt::getFalse(Context);
  Value *Call = Builder.CreateCall(ModelCastFunction,
                                   { SerializedTypeAsString,
                                     Operand,
                                     IsImplicit });
  return Call;
}

void MMCP::createAndInjectModelCast(Instruction *Ins,
                                    const SerializedType &ST,
                                    OpaqueFunctionsPool<TypePair> &Pool) {
  IRBuilder<> Builder(Ins);

  uint64_t OperandId = ST.OperandId;
  Value *Operand = Ins->getOperand(OperandId);
  Type *BaseAddressTy = Ins->getOperand(OperandId)->getType();
  Value *CallToModelCast = createCallToModelCast(Builder,
                                                 { BaseAddressTy,
                                                   BaseAddressTy },
                                                 ST.StringType,
                                                 Operand,
                                                 Pool);
  Ins->setOperand(OperandId, CallToModelCast);
}

bool MMCP::runOnFunction(Function &F) {
  bool Changed = false;

  Module *M = F.getParent();
  auto ModelCastPool = FunctionTags::ModelCast.getPool(*M);

  auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();
  const TupleTree<model::Binary> &Model = ModelWrapper.getReadOnlyModel();

  ModelFunction = llvmToModelFunction(*Model, F);
  revng_assert(ModelFunction != nullptr);

  // First of all, remove all SExt, ZExt and Trunc, and replace them with
  // ModelCasts.
  {
    LLVMContext &LLVMCtxt = F.getContext();
    IRBuilder<> Builder(LLVMCtxt);
    for (BasicBlock &BB : F) {
      for (Instruction &I : llvm::make_early_inc_range(BB)) {
        auto *SExt = dyn_cast<llvm::SExtInst>(&I);
        auto *ZExt = dyn_cast<llvm::ZExtInst>(&I);
        auto *Trunc = dyn_cast<llvm::TruncInst>(&I);
        if (not SExt and not ZExt and not Trunc)
          continue;

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

        // Create a string constant to pass as first argument of the call to
        // ModelCast, to represent the target model type.
        Constant *TargetModelTypeString = toLLVMString(ResultModelType, *M);
        revng_assert(TargetModelTypeString);

        Value *CallToModelCast = createCallToModelCast(Builder,
                                                       Key,
                                                       TargetModelTypeString,
                                                       CastedOperand,
                                                       ModelCastPool);
        I.replaceAllUsesWith(CallToModelCast);
        I.eraseFromParent();
      }
    }
  }

  TypeMap = initModelTypes(F, ModelFunction, *Model, false);

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto SerializedTypes = serializeTypesForModelCast(&I, *Model);

      if (SerializedTypes.empty())
        continue;

      Changed = true;

      for (unsigned Idx = 0; Idx < SerializedTypes.size(); ++Idx)
        createAndInjectModelCast(&I, SerializedTypes[Idx], ModelCastPool);
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
