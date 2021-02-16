#pragma once

//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/Pass.h"

struct LegacyPMAdjustStackPointerPass : public llvm::FunctionPass {
public:
  static char ID;

  LegacyPMAdjustStackPointerPass() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};
