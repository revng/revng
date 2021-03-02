#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

#include "revng/StackAnalysis/StackAnalysis.h"

namespace StackAnalysis {

class FunctionBoundariesDetectionPass : public llvm::ModulePass {
public:
  static char ID;

public:
  FunctionBoundariesDetectionPass() : llvm::ModulePass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
    AU.addRequired<StackAnalysis<true>>();
  }

  void serialize(std::ostream &Output, llvm::Module &M);

  bool runOnModule(llvm::Module &M) override;
};

} // namespace StackAnalysis
