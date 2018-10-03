#ifndef COLLECTNORETURN_H
#define COLLECTNORETURN_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <fstream>
#include <vector>

// LLVM includes
#include "llvm/Pass.h"

namespace llvm {
class BasicBlock;
}

class CollectNoreturn : public llvm::FunctionPass {
public:
  static char ID;

public:
  CollectNoreturn() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  void serialize(std::ostream &Output);

private:
  std::vector<llvm::BasicBlock *> NoreturnBBs;
};

#endif // COLLECTNORETURN_H
