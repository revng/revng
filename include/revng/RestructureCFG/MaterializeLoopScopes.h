#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

class MaterializeLoopScopes : public llvm::FunctionPass {
public:
  static char ID;

public:
  MaterializeLoopScopes() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};
