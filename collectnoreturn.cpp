/// \file collectnoreturn.cpp
/// \brief Implementation of the pass to collect the list of noreturn basic
///        blocks

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

// Local includes
#include "collectnoreturn.h"

using namespace llvm;

char CollectNoreturn::ID = 0;
static RegisterPass<CollectNoreturn> X("cnoreturn",
                                       "Collect noreturn Pass",
                                       true,
                                       true);

void CollectNoreturn::serialize(std::ostream &Output) {
  Output << "noreturn\n";
  for (BasicBlock *BB : NoreturnBBs)
    Output << BB->getName().data() << "\n";
}

bool CollectNoreturn::runOnFunction(Function &F) {
  NoreturnBBs.clear();

  for (BasicBlock &BB : F) {
    if (!BB.empty()) {
      TerminatorInst *Terminator = BB.getTerminator();
      if (Terminator->getMetadata("noreturn") != nullptr)
        NoreturnBBs.push_back(&BB);
    }
  }

  return false;
}
