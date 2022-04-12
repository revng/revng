#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/Pass.h"

#include "revng-c/RestructureCFGPass/RegionCFGTreeBB.h"

class RestructureCFG : public llvm::FunctionPass {

public:
  static char ID;

public:
  RestructureCFG() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};
