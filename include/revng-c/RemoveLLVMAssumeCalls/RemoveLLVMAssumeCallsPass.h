#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/Pass.h"

class RemoveLLVMAssumeCallsPass : public llvm::FunctionPass {
public:
  static char ID;

public:
  RemoveLLVMAssumeCallsPass() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};
