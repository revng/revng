#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"

class ABIAnalysisPass : public llvm::FunctionPass {
public:
  static char ID;

public:
  ABIAnalysisPass() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
  }
};
