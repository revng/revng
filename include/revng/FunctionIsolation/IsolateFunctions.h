#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>

#include "llvm/Pass.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Model/LoadModelPass.h"

class IsolateFunctions : public pipeline::ModulePass {
public:
  static char ID;

public:
  IsolateFunctions() : pipeline::ModulePass(ID) {}

  bool run(llvm::Module &M, const pipeline::TargetsList &Targets) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};
