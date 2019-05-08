/// \file RemoveUnexpectedPCPass.cpp
/// \brief FunctionPass that applies the comb to the RegionCFG of a function

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <sstream>
#include <stdlib.h>

// LLVM includes
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"

// revng includes
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

// Local libraries includes
#include "revng-c/RemoveUnexpectedPCPass/RemoveUnexpectedPCPass.h"

using namespace llvm;

char RemoveUnexpectedPC::ID = 0;
static RegisterPass<RemoveUnexpectedPC> X("remove-unexpected-pc",
                                          "Remove unexpectedpc from switches",
                                          false,
                                          false);

bool RemoveUnexpectedPC::runOnFunction(Function &F) {
  if (!F.getName().startswith("bb.")) {
    return false;
  }

  ReversePostOrderTraversal<llvm::Function *> RPOT(&F);

  for (BasicBlock *BB : RPOT) {
    LLVMContext &Context = getContext(&F);

    TerminatorInst *Terminator = BB->getTerminator();
    if (auto *Switch = dyn_cast<SwitchInst>(Terminator)) {

      // Get a pointer to the last successor basic block.
      revng_assert(Switch->getNumCases() > 0);
      unsigned SuccessorNumber = Switch->getNumSuccessors();
      BasicBlock *LastSuccessor = Switch->getSuccessor(SuccessorNumber - 1);

      // Remove the last successor from the cases.
      ConstantInt *LastCase = Switch->findCaseDest(LastSuccessor);
      Switch->removeCase(Switch->findCaseValue(LastCase));

      // Make the last successor the default case.
      Switch->setDefaultDest(LastSuccessor);
    }
  }

  removeUnreachableBlocks(F);

  return true;
}
