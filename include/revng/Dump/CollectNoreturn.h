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

class CollectNoreturn : public llvm::ModulePass {
public:
  static char ID;

public:
  CollectNoreturn() : llvm::ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  void serialize(std::ostream &Output);

private:
  std::vector<llvm::BasicBlock *> NoreturnBBs;
};

#endif // COLLECTNORETURN_H
