#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

#include "revng/Lift/LoadBinaryPass.h"
#include "revng/Model/LoadModelPass.h"

struct MakeSegmentRefPass : public llvm::ModulePass {
public:
  static char ID;

  MakeSegmentRefPass() : ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};
