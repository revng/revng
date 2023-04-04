#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
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
