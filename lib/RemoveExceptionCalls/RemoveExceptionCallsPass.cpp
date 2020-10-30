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
#include "revng-c/RemoveExceptionCalls/RemoveExceptionCallsPass.h"

using namespace llvm;

char RemoveExceptionCallsPass::ID = 0;
using Reg = RegisterPass<RemoveExceptionCallsPass>;
static Reg X("remove-exception-calls",
             "Removes calls to raise_exception_helper",
             true,
             true);

bool RemoveExceptionCallsPass::runOnFunction(Function &F) {

  // Skip non translated functions.
  if (not F.hasMetadata("revng.func.entry"))
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
