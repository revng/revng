#pragma once

//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/Pass.h"

#include "revng-c/DataLayoutAnalysis/DLALayouts.h"

struct AddPrimitiveTypesPass : public llvm::ModulePass {
  static char ID;

  AddPrimitiveTypesPass() : llvm::ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};
