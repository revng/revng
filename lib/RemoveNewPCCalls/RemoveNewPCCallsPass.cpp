//
// Copyright (c) rev.ng Srls 2017-2020.
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

  // Skip non translated functions.
  if (not F.hasMetadata("revng.func.entry"))
    return false;

  // Remove calls to `newpc` in the current function.
  SmallVector<Instruction *, 8> ToErase;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB)
      if (auto *C = dyn_cast<CallInst>(&I))
        if (getCallee(C)->getName() == "newpc")
          ToErase.push_back(C);
  }

  bool Changed = not ToErase.empty();
  for (Instruction *I : ToErase)
    I->eraseFromParent();

  return Changed;
}
