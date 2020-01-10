#pragma once

//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include <memory>

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Pass.h"
#include "llvm/PassAnalysisSupport.h"

struct DLAPass : public llvm::ModulePass {
  static char ID;

  DLAPass() : llvm::ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<llvm::LoopInfoWrapperPass>();
    AU.addRequired<llvm::ScalarEvolutionWrapperPass>();
    AU.setPreservesAll();
  }
};
