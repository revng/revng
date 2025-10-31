#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

class CPULoopExitPass : public llvm::ModulePass {
public:
  static char ID;

  CPULoopExitPass() : llvm::ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override;
};
