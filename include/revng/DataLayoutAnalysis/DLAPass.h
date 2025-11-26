#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Pass.h"

#include "revng/Model/Binary.h"

struct DLAPass : public llvm::ModulePass {
private:
  TupleTree<model::Binary> *ConstructorModel = nullptr;

public:
  static char ID;

  DLAPass() : llvm::ModulePass(ID) {}
  DLAPass(TupleTree<model::Binary> &Binary) :
    llvm::ModulePass(ID), ConstructorModel(&Binary) {}

  bool runOnModule(llvm::Module &M) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};
