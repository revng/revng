#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>

#include "llvm/Pass.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Model/LoadModelPass.h"

class InvokeIsolatedFunctionsPass : public llvm::ModulePass {
public:
  static char ID;

public:
  InvokeIsolatedFunctionsPass() : llvm::ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
    AU.addRequired<LoadModelWrapperPass>();
    AU.addRequired<pipeline::LoadExecutionContextPass>();
    AU.setPreservesAll();
  }
};
