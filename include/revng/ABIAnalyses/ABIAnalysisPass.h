#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
namespace ABIAnalyses {

class ABIAnalysisPass : public llvm::PassInfoMixin<ABIAnalysisPass> {
public:
  llvm::PreservedAnalyses
  run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);
};

class ABIAnalysisWrapperPass : public llvm::FunctionPass {
public:
  static char ID;

public:
  ABIAnalysisWrapperPass() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
  }
};

}
