#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/Pass.h"

class RemoveExceptionalCalls : public llvm::ModulePass {
public:
  static char ID;

public:
  RemoveExceptionalCalls() : llvm::ModulePass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  bool runOnModule(llvm::Module &F) override;
};
