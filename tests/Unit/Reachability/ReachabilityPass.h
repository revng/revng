#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <map>
#include <set>

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
