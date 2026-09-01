#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/PassManager.h"

#include "revng/Support/OpaqueFunctionsPool.h"

namespace llvm {
class GlobalVariable;
}

class RemoveHelperCallsPass
  : public llvm::PassInfoMixin<RemoveHelperCallsPass> {
  llvm::GlobalVariable *StackPointer = nullptr;

public:
  explicit RemoveHelperCallsPass(llvm::GlobalVariable *StackPointer) :
    StackPointer(StackPointer) {}

  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};
