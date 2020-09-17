#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

class DebugInfo : public llvm::ModulePass {
public:
  static char ID;

public:
  DebugInfo() : ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override;
};
