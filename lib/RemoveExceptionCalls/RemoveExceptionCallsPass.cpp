//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/IsolatedFunctions/IsolatedFunctions.h"
#include "revng-c/RemoveExceptionCalls/RemoveExceptionCallsPass.h"

using namespace llvm;

using RECPass = RemoveExceptionCallsPass;

char RECPass::ID = 0;
using Reg = RegisterPass<RECPass>;
static Reg X("remove-exception-calls",
             "Removes calls to raise_exception_helper",
             true,
             true);

void RECPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<LoadModelPass>();
}

bool RemoveExceptionCallsPass::runOnFunction(Function &F) {

  // Skip non-isolated functions
  const model::Binary &Model = getAnalysis<LoadModelPass>().getReadOnlyModel();
  if (not hasIsolatedFunction(Model, F))
    return false;

  // Remove calls to `raise_exception_helper` in the current function.
  SmallVector<Instruction *, 8> ToErase;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB)
      if (auto *C = dyn_cast<CallInst>(&I))
        if (getCallee(C)->getName() == "raise_exception_helper")
          ToErase.push_back(C);
  }

  bool Changed = not ToErase.empty();
  for (Instruction *I : ToErase)
    I->eraseFromParent();

  return Changed;
}
