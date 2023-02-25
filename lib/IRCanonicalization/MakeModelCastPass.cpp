//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Use.h"
#include "llvm/Pass.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"
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

using namespace llvm;
using ModelTypesMap = std::map<const llvm::Value *, const model::QualifiedType>;

struct SerializedType {
  Constant *StringType;
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
    AU.addRequired<FunctionMetadataCachePass>();
  }

private:
  std::vector<SerializedType>
  serializeTypesForModelCast(FunctionMetadataCache &Cache,
                             Instruction *,
                             const model::Binary &);
  void createAndInjectModelCast(Instruction *,
                                const SerializedType &,
                                OpaqueFunctionsPool<Type *> &);
};

using MMCP = MakeModelCastPass;

std::vector<SerializedType>
MMCP::serializeTypesForModelCast(FunctionMetadataCache &Cache,
                                 Instruction *I,
                                 const model::Binary &Model) {
  using namespace model;
  using namespace abi::FunctionType;

  std::vector<SerializedType> Result;
  Module *M = I->getModule();

  auto SerializeTypeFor =
    [this, &Model, &Result, &M, &Cache](const llvm::Use &Op) {
      // Check if we have strong model information about this operand
      auto ModelTypes = getExpectedModelType(Cache, &Op, Model);

      // Aggregates that do not correspond to model structs (e.g. return types
      // of RawFunctionTypes that return more than one value) cannot be handled
      // with casts, since we don't have a model::Type to cast them to.
      if (ModelTypes.size() == 1) {
        QualifiedType ExpectedType = ModelTypes.back();
        revng_assert(ExpectedType.UnqualifiedType().isValid());

        const QualifiedType &OperandType = TypeMap.at(Op.get());
        if (ExpectedType != OperandType) {
          revng_assert(ExpectedType.isScalar() and OperandType.isScalar());
          // Create a cast only if the expected type is different from the
          // actual type propagated until here
          auto Type = SerializedType(serializeToLLVMString(ExpectedType, *M),
                                     Op.getOperandNo());
          Result.emplace_back(std::move(Type));
        }
      }
    };

  if (auto *Call = dyn_cast<CallInst>(I)) {
    // Lifted functions have their prototype on the model
    auto *Callee = Call->getCalledFunction();

    if (FunctionTags::CallToLifted.isTagOf(Call)) {
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

  } else if (auto *SI = dyn_cast<StoreInst>(I)) {
    auto &PtrOperandPtrType = TypeMap.at(SI->getPointerOperand());
    auto &ValOperandType = TypeMap.at(SI->getValueOperand());

    const model::Architecture::Values &Arch = Model.Architecture();
    QualifiedType ValOperandPtrType = ValOperandType.getPointerTo(Arch);
    if (PtrOperandPtrType != ValOperandPtrType) {
      bool IsPtrOperandOfAggregateType = false;
      QualifiedType PtrOperandType;

      if (PtrOperandPtrType.isPointer()) {
        PtrOperandType = dropPointer(PtrOperandPtrType);
        const auto *Unqualified = PtrOperandType.UnqualifiedType().getConst();
        IsPtrOperandOfAggregateType = isa<model::StructType>(Unqualified)
                                      || isa<model::UnionType>(Unqualified);
      }

      if (isa<ConstantPointerNull>(SI->getPointerOperand())
          || IsPtrOperandOfAggregateType) {
        // Pointer operand needs to be casted to the model::QualifiedType of
        // the value operand, added pointer qualified.
        auto Type = SerializedType(serializeToLLVMString(ValOperandPtrType, *M),
                                   SI->getPointerOperandIndex());
        Result.emplace_back(std::move(Type));
      } else {
        // Value operand needs to be casted to the drop'd ptr of the
        // model::QualifiedType of the pointer type.
        revng_assert(PtrOperandPtrType.isPointer());
        auto *LLVMString = serializeToLLVMString(PtrOperandType, *M);
        auto Type = SerializedType(LLVMString);
        Result.emplace_back(std::move(Type));
      }
    }
  }

  return Result;
}

void MMCP::createAndInjectModelCast(Instruction *Ins,
                                    const SerializedType &ST,
                                    OpaqueFunctionsPool<Type *> &Pool) {
  IRBuilder<> Builder(Ins);

  uint64_t OperandId = ST.OperandId;
  Constant *StringType = ST.StringType;
  Type *BaseAddressTy = Ins->getOperand(OperandId)->getType();

  llvm::Type *StringPtrType = getStringPtrType(Ins->getContext());
  auto *ModelCastType = FunctionType::get(BaseAddressTy,
                                          { StringPtrType, BaseAddressTy },
                                          false);
  auto *ModelCastFunction = Pool.get(BaseAddressTy, ModelCastType, "ModelCast");

  Value *Call = Builder.CreateCall(ModelCastFunction,
                                   { StringType, Ins->getOperand(OperandId) });
  Ins->setOperand(OperandId, Call);
}

bool MMCP::runOnFunction(Function &F) {
  bool Changed = false;

  OpaqueFunctionsPool<Type *> ModelCastPool(F.getParent(), false);
  initModelCastPool(ModelCastPool);

  auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();
  const TupleTree<model::Binary> &Model = ModelWrapper.getReadOnlyModel();

  ModelFunction = llvmToModelFunction(*Model, F);
  revng_assert(ModelFunction != nullptr);
  auto &Cache = getAnalysis<FunctionMetadataCachePass>().get();

  TypeMap = initModelTypes(Cache, F, ModelFunction, *Model, false);

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto SerializedTypes = serializeTypesForModelCast(Cache, &I, *Model);

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
