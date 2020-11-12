#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/Pass.h"

class RemoveCpuLoopStorePass : public llvm::FunctionPass {
public:
  static char ID;

public:
  RemoveCpuLoopStorePass() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;
};
