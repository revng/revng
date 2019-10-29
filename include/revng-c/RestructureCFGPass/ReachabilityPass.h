#ifndef REVNGC_RESTRUCTURE_CFG_REACHABILITY_H
#define REVNGC_RESTRUCTURE_CFG_REACHABILITY_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <fstream>

// LLVM includes
#include "llvm/Pass.h"

class ReachabilityPass : public llvm::FunctionPass {
public:
  static char ID;

public:
  ReachabilityPass() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  bool existsPath(llvm::BasicBlock *Source, llvm::BasicBlock *Target);
  std::set<llvm::BasicBlock *> &reachableFrom(llvm::BasicBlock *Source);

private:
  std::map<llvm::BasicBlock *, std::set<llvm::BasicBlock *>> ReachableBlocks;
};

#endif // REVNGC_RESTRUCTURE_CFG_REACHABILITY_H
