//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

// revng includes
#include <revng/Support/IRHelpers.h>

// Local libraries includes
#include "revng-c/RemoveCpuLoopStore/RemoveCpuLoopStorePass.h"

using namespace llvm;

char RemoveCpuLoopStorePass::ID = 0;
using Reg = RegisterPass<RemoveCpuLoopStorePass>;
static Reg X("remove-cpu-loop-store", "Removes store to cpu_loop", true, true);

bool RemoveCpuLoopStorePass::runOnFunction(Function &F) {
  if (not F.hasMetadata("revng.func.entry"))
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
