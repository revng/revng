//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/RemoveNewPCCalls/RemoveNewPCCallsPass.h"

using namespace llvm;

char RemoveNewPCCallsPass::ID = 0;
using Reg = RegisterPass<RemoveNewPCCallsPass>;
static Reg X("remove-newpc-calls", "Removes calls to newpc", true, true);

void RemoveNewPCCallsPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<LoadModelWrapperPass>();
}

bool RemoveNewPCCallsPass::runOnFunction(Function &F) {

  // Skip non-isolated functions
  auto FTags = FunctionTags::TagsSet::from(&F);
  if (not FTags.contains(FunctionTags::Lifted))
    return false;

  // Remove calls to `newpc` in the current function.
  SmallVector<Instruction *, 8> ToErase;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB)
      if (auto *C = dyn_cast<CallInst>(&I))
        if (auto *Callee = getCallee(C);
            Callee and Callee->getName() == "newpc")
          ToErase.push_back(C);
  }

  bool Changed = not ToErase.empty();
  for (Instruction *I : ToErase)
    eraseFromParent(I);

  return Changed;
}
