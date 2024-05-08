#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/IR/PassManager.h"

class RemoveNewPCCallsPass : public llvm::PassInfoMixin<RemoveNewPCCallsPass> {

public:
  RemoveNewPCCallsPass() = default;

  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};
