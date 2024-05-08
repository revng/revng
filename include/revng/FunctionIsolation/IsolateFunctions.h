#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <memory>

#include "llvm/Pass.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Model/LoadModelPass.h"

class IsolateFunctions : public llvm::ModulePass {
public:
  static char ID;

public:
  IsolateFunctions() : ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};
