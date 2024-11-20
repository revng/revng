#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

/// Enrich StackOffsetMarker-tagged calls with LazyValueInfo boundaries info
struct ComputeStackAccessesBoundsPass : public llvm::FunctionPass {
public:
  static char ID;

  ComputeStackAccessesBoundsPass() : llvm::FunctionPass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool runOnFunction(llvm::Function &F) override;
};
