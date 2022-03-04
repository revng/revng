#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/Pass.h"

class RestoreExtractValues : public llvm::FunctionPass {
public:
  static char ID;

public:
  RestoreExtractValues() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &) override;
};
