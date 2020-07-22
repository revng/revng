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
#include "revng-c/RemoveNewPCCalls/RemoveNewPCCallsPass.h"

using namespace llvm;

char RemoveNewPCCallsPass::ID = 0;
using Reg = RegisterPass<RemoveNewPCCallsPass>;
static Reg X("remove-newpc-calls", "Removes calls to newpc", true, true);

bool RemoveNewPCCallsPass::runOnFunction(Function &F) {
  bool Changed = false;
  // Remove calls to newpc in isolated functions
  for (Function &ParentF : *F.getParent()) {
    for (BasicBlock &BB : ParentF) {
      if (not ParentF.hasMetadata("revng.func.entry"))
        continue;

      SmallVector<Instruction *, 8> ToErase;
      for (Instruction &I : BB)
        if (auto *C = dyn_cast<CallInst>(&I))
          if (getCallee(C)->getName() == "newpc")
            ToErase.push_back(C);

      Changed |= not ToErase.empty();
      for (Instruction *I : ToErase)
        I->eraseFromParent();
    }
  }
  return Changed;
}
