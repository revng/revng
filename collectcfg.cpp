/// \file collectcfg.cpp
/// \brief Implementation of the pass to collect the CFG in a readable format.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

// Local includes
#include "collectcfg.h"
#include "datastructures.h"
#include "ir-helpers.h"

using namespace llvm;

char CollectCFG::ID = 0;
static RegisterPass<CollectCFG> X("ccfg", "Collect CFG Pass", true, true);

void CollectCFG::serialize(std::ostream &Output) {
  Output << "source,destination\n";
  for (auto &P : Result) {
    BasicBlock *Source = P.first;
    std::sort(P.second.begin(), P.second.end(), CompareByName<BasicBlock>());
    for (BasicBlock *Destination : P.second)
      Output << Source->getName().data() << "," << Destination->getName().data()
             << "\n";
  }
}

bool CollectCFG::isNewInstruction(BasicBlock *BB) {
  if (BB->empty())
    return false;

  auto *Call = dyn_cast<CallInst>(&*BB->begin());
  if (Call == nullptr || Call->getCalledFunction() == nullptr
      || Call->getCalledFunction()->getName() != "newpc")
    return false;

  return true;
}

bool CollectCFG::runOnFunction(Function &F) {
  Result.clear();

  for (BasicBlock &BB : F) {
    if (!isNewInstruction(&BB))
      BlackList.insert(&BB);
    else
      break;
  }

  // For each basic block
  for (BasicBlock &BB : F) {
    if (!isNewInstruction(&BB))
      continue;

    OnceQueue<BasicBlock *> Queue;
    Queue.insert(&BB);
    while (!Queue.empty()) {
      BasicBlock *ToExplore = Queue.pop();
      for (BasicBlock *Successor : successors(ToExplore)) {

        // If it's a new instruction register it, otherwise enqueue the basic
        // block for further processing
        if (isNewInstruction(Successor)) {
          Result[&BB].push_back(Successor);
        } else if (BlackList.count(Successor) == 0) {
          Queue.insert(Successor);
        }
      }
    }
  }

  return false;
}
