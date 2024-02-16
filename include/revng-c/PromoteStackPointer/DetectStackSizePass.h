#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

/// Import into the model information about stack arguments/frame sizes
struct DetectStackSizePass : public llvm::ModulePass {
public:
  static char ID;

  DetectStackSizePass() : llvm::ModulePass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool runOnModule(llvm::Module &M) override;
};
