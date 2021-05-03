#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"

class InlineHelpersPass : public llvm::FunctionPass {
public:
  static char ID;

public:
  InlineHelpersPass() : FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;
};
