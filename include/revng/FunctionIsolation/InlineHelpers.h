#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/Pass.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"

class InlineHelpersPass : public llvm::ModulePass {
public:
  static char ID;

public:
  InlineHelpersPass() : ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override;
};
