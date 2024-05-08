#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/Pass.h"

class RemoveDbgMetadata : public llvm::FunctionPass {
public:
  static char ID;

public:
  RemoveDbgMetadata() : llvm::FunctionPass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  bool runOnFunction(llvm::Function &F) override;
};
