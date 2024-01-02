//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

class RemoveLLVMAssumeCallsPass : public llvm::FunctionPass {
public:
  static char ID;

public:
  RemoveLLVMAssumeCallsPass() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};

using RemoveAssumePass = RemoveLLVMAssumeCallsPass;

char RemoveAssumePass::ID = 0;
using Reg = RegisterPass<RemoveAssumePass>;
static Reg
  X("remove-llvmassume-calls", "Removes calls to assume intrinsic", true, true);

void RemoveAssumePass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
}

bool RemoveAssumePass::runOnFunction(Function &F) {

  // Skip non-isolated functions
  auto FTags = FunctionTags::TagsSet::from(&F);
  if (not FTags.contains(FunctionTags::Isolated))
    return false;

  // Remove calls to `llvm.assume` in isolated functions.
  SmallVector<Instruction *, 8> ToErase;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB)
      if (auto *C = dyn_cast<CallInst>(&I))
        if (auto *Callee = getCallee(C);
            Callee and Callee->getName() == "llvm.assume")
          ToErase.push_back(C);
  }

  bool Changed = not ToErase.empty();
  for (Instruction *I : ToErase)
    eraseFromParent(I);

  return Changed;
}
