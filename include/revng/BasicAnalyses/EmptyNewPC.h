#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/Pass.h"

class EmptyNewPC : public llvm::ModulePass {
public:
  static char ID;

public:
  EmptyNewPC() : llvm::ModulePass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  bool runOnModule(llvm::Module &M) override;
};
