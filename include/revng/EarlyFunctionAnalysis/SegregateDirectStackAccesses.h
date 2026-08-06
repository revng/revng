#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/PassManager.h"

namespace llvm {
class GlobalVariable;
}

class SegregateDirectStackAccessesPass
  : public llvm::PassInfoMixin<SegregateDirectStackAccessesPass> {
private:
  llvm::GlobalVariable *StackPointer = nullptr;

public:
  explicit SegregateDirectStackAccessesPass(llvm::GlobalVariable
                                              *StackPointer) :
    StackPointer(StackPointer) {}

  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};
