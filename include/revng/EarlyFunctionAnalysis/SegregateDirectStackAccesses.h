#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/PassManager.h"

#include "revng/Support/revng.h"

class SegregateDirectStackAccessesPass
  : public llvm::PassInfoMixin<SegregateDirectStackAccessesPass> {

public:
  llvm::PreservedAnalyses
  run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);
};
