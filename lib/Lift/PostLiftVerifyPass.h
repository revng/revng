#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

class PostLiftVerifyPass : public llvm::ModulePass {
public:
  static char ID;

public:
  PostLiftVerifyPass() : llvm::ModulePass(ID) {}

public:
  bool runOnModule(llvm::Module &M) final;
};
