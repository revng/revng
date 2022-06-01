//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Pass.h"

#include "revng/ABI/FunctionType.h"
#include "revng/EarlyFunctionAnalysis/IRHelpers.h"
#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"

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
  }

private:
  std::vector<SerializedType>
  serializeTypesForModelCast(Instruction *, const model::Binary &);
  void createAndInjectModelCast(Instruction *,
                                const SerializedType &,
                                OpaqueFunctionsPool<Type *> &);
};

using MMCP = MakeModelCastPass;

std::vector<SerializedType>
MMCP::serializeTypesForModelCast(Instruction *I, const model::Binary &Model) {
  using namespace model;
  using namespace abi::FunctionType;

  std::vector<SerializedType> Result;
  Module *M = I->getModule();
  const model::Type *ParentPrototype = ModelFunction->Prototype.get();

  if (auto *CI = dyn_cast<CallInst>(I)) {
    if (auto *Callee = CI->getCalledFunction()) {
      // Do we have a ModelGEP which has its base llvm::Value embedding a cast?
      if (FunctionTags::ModelGEP.isTagOf(Callee)) {
        auto &PtrType = TypeMap.at(CI->getArgOperand(1));
        auto CurTypePtr = deserializeAndParseQualifiedType(CI->getArgOperand(0),
                                                           Model);
        addPointerQualifier(CurTypePtr, Model);

        if (PtrType != CurTypePtr) {
          auto Type = SerializedType(serializeToLLVMString(CurTypePtr, *M), 1);
          Result.emplace_back(std::move(Type));
        }
      } else if (FunctionTags::StructInitializer.isTagOf(Callee)) {
        // Do we have a function that return multiple return values and may not
        // match function's return type?
        auto *RFT = cast<RawFunctionType>(ParentPrototype);
        auto &FormalRetVals = RFT->ReturnValues;

        for (const auto &[LLVMArg, RetVal] :
             llvm::zip(CI->args(), FormalRetVals)) {
          auto &StructInitializerArgType = TypeMap.at(LLVMArg);
          auto FormalRetValType = RetVal.Type;

          if (StructInitializerArgType != FormalRetValType) {
            auto Type = SerializedType(serializeToLLVMString(FormalRetValType,
                                                             *M),
                                       LLVMArg.getOperandNo());
            Result.emplace_back(std::move(Type));
          }
        }
      } else if (auto *ModelFunctionCallee = llvmToModelFunction(Model,
                                                                 *Callee)) {
        // Do we have a function call with a cast on one of its argument?
        auto Layout = Layout::make(ModelFunctionCallee->Prototype);
        for (const auto &Pair : llvm::enumerate(Layout.Arguments)) {
          auto &Arg = Pair.value();
          uint64_t OperandId = Pair.index();

          auto &ArgType = TypeMap.at(CI->getArgOperand(OperandId));

          // Stack argument type
          if (!Arg.Type.isScalar())
            addPointerQualifier(Arg.Type, Model);

          model::QualifiedType FormalArgType = Arg.Type;

          if (ArgType != FormalArgType) {
            auto Type = SerializedType(serializeToLLVMString(FormalArgType, *M),
                                       OperandId);
            Result.emplace_back(std::move(Type));
          }
        }
      }
    } else {
      // Indirect call
      const auto &PrototypePath = getCallSitePrototype(Model, CI);
      llvm::Value *CalledVal = CI->getCalledOperand();

      auto &FuncPtrType = TypeMap.at(CalledVal);
      QualifiedType FormalPtrType = createPointerTo(PrototypePath, Model);

      if (FuncPtrType != FormalPtrType) {
        auto Type = SerializedType(serializeToLLVMString(FormalPtrType, *M),
                                   CI->getCalledOperandUse().getOperandNo());
        Result.emplace_back(std::move(Type));
      }
    }
  } else if (auto *RI = dyn_cast<ReturnInst>(I)) {
    // Do we have a return instruction whose operand type is different from the
    // function type being returned? Multiple return values are already handled
    // in StructInitializer.
    if (RI->getNumOperands() != 0 && !isa<Instruction>(RI->getReturnValue())) {
      QualifiedType FuncRetType;

      if (auto *RFT = dyn_cast<RawFunctionType>(ParentPrototype)) {
        FuncRetType = RFT->ReturnValues.begin()->Type;
        revng_assert(RFT->ReturnValues.size() == 1);
      } else if (auto *CABIF = dyn_cast<CABIFunctionType>(ParentPrototype)) {
        FuncRetType = CABIF->ReturnType;
      } else {
        revng_abort("Unknown function type");
      }

      auto &RetValueType = TypeMap.at(RI->getOperand(0));
      if (RetValueType != FuncRetType) {
        auto Type = SerializedType(serializeToLLVMString(FuncRetType, *M));
        Result.emplace_back(std::move(Type));
      }
    }
  } else if (auto *SI = dyn_cast<StoreInst>(I)) {
    auto &PtrOperandPtrType = TypeMap.at(SI->getPointerOperand());
    auto &ValOperandType = TypeMap.at(SI->getValueOperand());

    QualifiedType ValOperandPtrType = ValOperandType;
    addPointerQualifier(ValOperandPtrType, Model);

    if (PtrOperandPtrType != ValOperandPtrType) {
      bool IsPtrOperandOfAggregateType = false;
      QualifiedType PtrOperandType;

      if (PtrOperandPtrType.isPointer()) {
        PtrOperandType = dropPointer(PtrOperandPtrType);
        const auto *Unqualified = PtrOperandType.UnqualifiedType.getConst();
        IsPtrOperandOfAggregateType = isa<model::StructType>(Unqualified)
                                      || isa<model::UnionType>(Unqualified);
      }

      if (isa<ConstantPointerNull>(SI->getPointerOperand())
          || IsPtrOperandOfAggregateType) {
        // Pointer operand needs to be casted to the model::QualifiedType of the
        // value operand, added pointer qualified.
        auto Type = SerializedType(serializeToLLVMString(ValOperandPtrType, *M),
                                   SI->getPointerOperandIndex());
        Result.emplace_back(std::move(Type));
      } else {
        // Value operand needs to be casted to the drop'd ptr of the
        // model::QualifiedType of the pointer type.
        revng_assert(PtrOperandPtrType.isPointer());
        auto Type = SerializedType(serializeToLLVMString(PtrOperandType, *M));
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

  auto *ModelCastType = FunctionType::get(BaseAddressTy,
                                          { getStringPtrType(Ins->getContext()),
                                            BaseAddressTy },
                                          false);
  auto *ModelCastFunction = Pool.get(BaseAddressTy, ModelCastType, "modelCast");

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

  TypeMap = initModelTypes(F, ModelFunction, *Model, false);

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto SerializedTypes = serializeTypesForModelCast(&I, *Model);

      if (!SerializedTypes.empty()) {
        Changed = true;

        for (unsigned Idx = 0; Idx < SerializedTypes.size(); ++Idx)
          createAndInjectModelCast(&I, SerializedTypes[Idx], ModelCastPool);
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
