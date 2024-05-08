#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/IR/PassManager.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Support/OpaqueFunctionsPool.h"

class RemoveHelperCallsPass
  : public llvm::PassInfoMixin<RemoveHelperCallsPass> {
  GeneratedCodeBasicInfo *GCBI = nullptr;

public:
  RemoveHelperCallsPass() = default;

  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};
