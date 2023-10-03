#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/Pass.h"

class RemoveExtractValues : public llvm::FunctionPass {
public:
  static char ID;

public:
  RemoveExtractValues() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &) override;
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};
