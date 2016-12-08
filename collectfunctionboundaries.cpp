/// \file collectfunctionboundaries.cpp
/// \brief Implementation of the pass to collect the function boundaries

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"

// Local includes
#include "collectfunctionboundaries.h"

using namespace llvm;

template<typename T>
struct CompareByName {
  bool operator()(const T *LHS, const T *RHS) const {
    return LHS->getName() < RHS->getName();
  }
};

char CollectFunctionBoundaries::ID = 0;
static RegisterPass<CollectFunctionBoundaries> X("cfb",
                                                 "Collect function boundaries "
                                                 "Pass",
                                                 true,
                                                 true);

void CollectFunctionBoundaries::serialize(std::ostream &Output) {
  Output << "function,basicblock\n";

  auto Comparator = CompareByName<const BasicBlock>();
  for (auto &P : Functions) {
    std::sort(P.second.begin(), P.second.end(), Comparator);
    for (BasicBlock *BB : P.second) {
      Output << P.first.data() << "," << BB->getName().data() << "\n";
    }
  }
}

bool CollectFunctionBoundaries::runOnFunction(Function &F) {
  Functions.clear();

  for (BasicBlock &BB : F) {
    if (!BB.empty()) {
      TerminatorInst *Terminator = BB.getTerminator();
      if (MDNode *Node = Terminator->getMetadata("func")) {
        auto *Tuple = cast<MDTuple>(Node);
        for (const MDOperand &Op : Tuple->operands()) {
          auto *FunctionMD = cast<MDTuple>(Op);
          auto *FunctionNameMD = cast<MDString>(&*FunctionMD->getOperand(0));
          Functions[FunctionNameMD->getString()].push_back(&BB);
        }
      }
    }
  }

  return false;
}
