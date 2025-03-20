#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

class SelectScopePass : public llvm::FunctionPass {
public:
  static char ID;

public:
  SelectScopePass() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};
