#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/Pass.h"

class RemoveExceptionCallsPass : public llvm::FunctionPass {
public:
  static char ID;

public:
  RemoveExceptionCallsPass() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;
};
