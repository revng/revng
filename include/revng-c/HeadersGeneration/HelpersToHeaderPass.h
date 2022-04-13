#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/Pass.h"

/// This pass collects all the calls to QEMU helpers and
/// generates a C declaration for each helper.
class HelpersToHeaderPass : public llvm::ModulePass {
public:
  static char ID;

  HelpersToHeaderPass() : llvm::ModulePass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool runOnModule(llvm::Module &M) override;
};
