/// \file collectcfg.cpp
/// \brief Implementation of the pass to collect the CFG in a readable format.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

// Local libraries includes
#include "revng/ADT/Queue.h"
#include "revng/Dump/CollectCFG.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;
using namespace llvm::cl;

char CollectCFG::ID = 0;
using RegisterCCFG = RegisterPass<CollectCFG>;
static RegisterCCFG X("collect-cfg", "Collect CFG Pass", true, true);

static opt<std::string> OutputPath("collect-cfg-output",
                                   desc("Destination path for the Collect CFG "
                                        "Pass"),
                                   value_desc("path"),
                                   cat(MainCategory));

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

bool CollectCFG::runOnModule(Module &M) {
  Function &F = *M.getFunction("root");
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

  if (OutputPath.getNumOccurrences() == 1) {
    std::ofstream Output;
    serialize(pathToStream(OutputPath, Output));
  }

  return false;
}
