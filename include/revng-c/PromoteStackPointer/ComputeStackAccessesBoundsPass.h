#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/Pass.h"

/// Enrich StackOffsetMarker-tagged calls with LazyValueInfo boundaries info
struct ComputeStackAccessesBoundsPass : public llvm::ModulePass {
public:
  static char ID;

  ComputeStackAccessesBoundsPass() : llvm::ModulePass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool runOnModule(llvm::Module &M) override;
};
