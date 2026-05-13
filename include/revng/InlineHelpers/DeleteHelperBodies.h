#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

class DeleteHelperBodiesPass : public llvm::ModulePass {
public:
  static char ID;

public:
  DeleteHelperBodiesPass() : llvm::ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override;
};
