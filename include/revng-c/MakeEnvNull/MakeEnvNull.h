#pragma once

//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/Pass.h"

struct MakeEnvNullPass : public llvm::ModulePass {
public:
  static char ID;

  MakeEnvNullPass() : llvm::ModulePass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  bool runOnModule(llvm::Module &F) override;
};
