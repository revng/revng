//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/IsolatedFunctions/IsolatedFunctions.h"
#include "revng-c/RemoveCpuLoopStore/RemoveCpuLoopStorePass.h"

using namespace llvm;

char RemoveCpuLoopStorePass::ID = 0;
using Reg = RegisterPass<RemoveCpuLoopStorePass>;
static Reg X("remove-cpu-loop-store", "Removes store to cpu_loop", true, true);

void RemoveCpuLoopStorePass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<LoadModelWrapperPass>();
}

bool RemoveCpuLoopStorePass::runOnFunction(Function &F) {

  // Skip non-isolated functions
  const model::Binary
    &Model = getAnalysis<LoadModelWrapperPass>().get().getReadOnlyModel();
  if (not hasIsolatedFunction(Model, F))
    return false;

  // Retrieve the global variable `cpu_loop_exiting`
  Module *M = F.getParent();
  GlobalVariable *CpuLoop = M->getGlobalVariable("cpu_loop_exiting");

  // Remove in bulk all the users of the global variable. We directly cast
  // the user to a `llvm::StoreInst` by design (if we have another kind of user
  // our assumptions does not hold.
  SmallVector<Instruction *, 8> ToErase;

  for (User *U : CpuLoop->users()) {
    Instruction *I = cast<Instruction>(U);

    // Check only translated functions.
    if (I->getParent()->getParent() != &F)
      continue;

    StoreInst *Store = cast<StoreInst>(U);
    ToErase.push_back(Store);
  }

  // Remove in bulk all the store found before.
  bool Changed = not ToErase.empty();
  for (Instruction *I : ToErase) {
    I->eraseFromParent();
  }

  return Changed;
}
