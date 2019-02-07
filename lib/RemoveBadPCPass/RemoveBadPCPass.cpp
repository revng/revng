/// \file Restructure.cpp
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
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

// revng includes
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

// Local libraries includes
#include "revng-c/RemoveBadPCPass/RemoveBadPCPass.h"

using namespace llvm;

char RemoveBadPC::ID = 0;
static RegisterPass<RemoveBadPC> X("removeBadPC",
                                   "Remove bad return pc check",
                                   false,
                                   false);

bool RemoveBadPC::runOnFunction(Function &F) {
  if (!F.getName().startswith("bb.")) {
    return false;
  }

  ReversePostOrderTraversal<llvm::Function *> RPOT(&F);

  for (BasicBlock *BB : RPOT) {

    // Identify the `_bad_return_pc` check nodes.
    if (BB->getName().endswith("_bad_return_pc")) {
      revng_assert(BB->getUniquePredecessor() != nullptr);


      // Get the basic block that performs the pc check.
      BasicBlock *Predecessor = BB->getUniquePredecessor();
      TerminatorInst *Terminator = Predecessor->getTerminator();
      BranchInst *Branch = cast<BranchInst>(Terminator);
      revng_assert(Branch->isConditional());

      // The `_bad_return_pc` basic block is the one corresponding to the else
      // branch, so we need to keep only the `then` branch.
      BasicBlock *Then = Branch->getSuccessor(0);
      BasicBlock *Else = Branch->getSuccessor(1);

      // Print the name of the block being removed.
      //dbg << "Removing successor: " << Else->getName().str() << "\n";

      // Remove the conditional branch.
      Branch->eraseFromParent();

      // Add the unconditional branch.
      BranchInst::Create(Then, Predecessor);

      // Remove the dandling basic block
      BB->eraseFromParent();
    }
  }

  return true;
}
