#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>

#include "llvm/Pass.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Model/LoadModelPass.h"

class InvokeIsolatedFunctionsPass : public pipeline::ModulePass {
public:
  static char ID;

public:
  InvokeIsolatedFunctionsPass() : pipeline::ModulePass(ID) {}

  bool run(llvm::Module &M, const pipeline::TargetsList &Targets) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
    AU.addRequired<LoadModelWrapperPass>();
    AU.addRequired<pipeline::LoadExecutionContextPass>();
    AU.setPreservesAll();
  }
};
