#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <memory>

#include "llvm/Pass.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"

class EnforceABI : public llvm::ModulePass {
public:
  static char ID;

public:
  EnforceABI() : ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};
