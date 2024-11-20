#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

/// Remove all temporary changes to the IR introduced to perform stack-related
/// computations
struct CleanupStackSizeMarkersPass : public llvm::ModulePass {
public:
  static char ID;

  CleanupStackSizeMarkersPass() : llvm::ModulePass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool runOnModule(llvm::Module &M) override;
};
