//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"

#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Support/OpaqueFunctionsPool.h"

#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/ModelHelpers.h"

static Logger<> Log{ "make-local-variables" };

struct MakeLocalVariables : public llvm::FunctionPass {
public:
  static char ID;

  MakeLocalVariables() : FunctionPass(ID) {}

  /// Transform allocas and alloca-like function calls into calls to
  /// `LocalVariable(AddressOf())`
  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<LoadModelWrapperPass>();
    AU.setPreservesCFG();
  }
};

using llvm::AllocaInst;
using llvm::dyn_cast;

bool MakeLocalVariables::runOnFunction(llvm::Function &F) {

  // Skip non-isolated functions
  auto FTags = FunctionTags::TagsSet::from(&F);
  if (not FTags.contains(FunctionTags::Isolated))
    return false;

  // Get the model
  const auto
    &Model = getAnalysis<LoadModelWrapperPass>().get().getReadOnlyModel().get();

  llvm::SmallVector<llvm::AllocaInst *, 8> ToReplace;

  // Collect instructions that allocate local variables
  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *Alloca = dyn_cast<AllocaInst>(&I))
        ToReplace.push_back(Alloca);

  if (ToReplace.empty())
    return false;

  llvm::LLVMContext &LLVMCtx = F.getContext();
  llvm::Module &M = *F.getParent();
  llvm::IRBuilder<> Builder(LLVMCtx);
  llvm::Type *PtrSizedInteger = getPointerSizedInteger(LLVMCtx, *Model);

  // Initialize function pools
  OpaqueFunctionsPool<TypePair> AddressOfPool(&M, false);
  initAddressOfPool(AddressOfPool, &M);
  OpaqueFunctionsPool<llvm::Type *> LocalVarPool(&M, false);
  initLocalVarPool(LocalVarPool);

  for (auto *Alloca : ToReplace) {
    Builder.SetInsertPoint(Alloca);
    llvm::Type *ResultType = Alloca->getType();

    // Convert the allocated llvm type to a model type
    auto AllocatedType = llvmIntToModelType(Alloca->getAllocatedType(), *Model);
    llvm::Constant *ModelTypeString = serializeToLLVMString(AllocatedType, M);

    auto LocalVarLLVMType = llvm::IntegerType::get(LLVMCtx,
                                                   AllocatedType.size().value()
                                                     * 8);

    // Inject call to LocalVariable
    auto *LocalVarFunctionType = getLocalVarType(LocalVarLLVMType);
    auto *LocalVarFunction = LocalVarPool.get(LocalVarLLVMType,
                                              LocalVarFunctionType,
                                              "LocalVariable");
    auto *LocalVarCall = Builder.CreateCall(LocalVarFunction,
                                            { ModelTypeString });

    // Inject a call to AddressOf
    auto LocalVarType = LocalVarCall->getType();
    auto *AddressOfFunctionType = getAddressOfType(PtrSizedInteger,
                                                   LocalVarType);
    auto *AddressOfFunction = AddressOfPool.get({ PtrSizedInteger,
                                                  LocalVarType },
                                                AddressOfFunctionType,
                                                "AddressOf");
    llvm::Instruction *AddressOfCall = Builder.CreateCall(AddressOfFunction,
                                                          { ModelTypeString,
                                                            LocalVarCall });

    AddressOfCall->copyMetadata(*Alloca);
    llvm::Value *ValueToSubstitute = AddressOfCall;

    // LocalVar and AddressOf work on LLVM integers that represent
    // pointers in the binary. If we are actually using these as
    // pointers in LLVM IR, we need to cast them to the appropriate
    // type.
    if (ResultType->isPointerTy())
      ValueToSubstitute = Builder.CreateIntToPtr(AddressOfCall, ResultType);

    revng_assert(ResultType == ValueToSubstitute->getType());

    Alloca->replaceAllUsesWith(ValueToSubstitute);
    Alloca->eraseFromParent();
  }

  return true;
}

char MakeLocalVariables::ID = 0;

static llvm::RegisterPass<MakeLocalVariables> X("make-local-variables",
                                                "Replace all opcodes "
                                                "that "
                                                "declare local "
                                                "variables with "
                                                "a call to "
                                                "LocalVariable.",
                                                false,
                                                false);
