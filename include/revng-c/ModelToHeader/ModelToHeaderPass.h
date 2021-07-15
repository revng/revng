#pragma once

//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/Pass.h"

class ModelToHeaderPass : public llvm::ModulePass {
public:
  static char ID;

  ModelToHeaderPass() : llvm::ModulePass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool runOnModule(llvm::Module &M) override;
};
