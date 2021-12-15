#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

#include "revng/EarlyFunctionAnalysis/EarlyFunctionAnalysis.h"

namespace EarlyFunctionAnalysis {

class ABIDetectionPass : public llvm::ModulePass {
public:
  static char ID;

public:
  ABIDetectionPass() : llvm::ModulePass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
    AU.addRequired<EarlyFunctionAnalysis>();
  }

  bool runOnModule(llvm::Module &M) override;
};

} // namespace EarlyFunctionAnalysis
