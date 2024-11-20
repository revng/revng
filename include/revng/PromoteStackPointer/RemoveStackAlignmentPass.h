#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

/// Drop stack alignment code from functions
///
/// We assume the stack is already aligned and there's no need to align it, for
/// analysis purposes. If we don't do this, we cannot correctly established the
/// stack offset w.r.t. the initial value of the stack pointer along the
/// function body.
struct RemoveStackAlignmentPass : public llvm::ModulePass {
public:
  static char ID;

  RemoveStackAlignmentPass() : llvm::ModulePass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool runOnModule(llvm::Module &M) override;
};
