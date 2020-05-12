#pragma once

//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/Pass.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"

struct PromoteStackPointerPass : public llvm::FunctionPass {
public:
  static char ID;

  PromoteStackPointerPass() : llvm::FunctionPass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
    AU.setPreservesCFG();
  }

  bool runOnFunction(llvm::Function &F) override;
};
