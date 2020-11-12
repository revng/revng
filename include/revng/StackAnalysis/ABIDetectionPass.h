#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

#include "revng/StackAnalysis/StackAnalysis.h"

namespace StackAnalysis {

class ABIDetectionPass : public llvm::ModulePass {
public:
  static char ID;

public:
  ABIDetectionPass() : llvm::ModulePass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<StackAnalysis<true>>();
  }

  bool runOnModule(llvm::Module &M) override;
};

} // namespace StackAnalysis
