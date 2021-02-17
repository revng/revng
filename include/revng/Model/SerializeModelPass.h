#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

#include "revng/Model/Binary.h"
#include "revng/Model/LoadModelPass.h"

class SerializeModelPass : public llvm::ModulePass {
public:
  static char ID;

public:
  SerializeModelPass() : llvm::ModulePass(ID) {}
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override final {
    AU.setPreservesAll();
    AU.addRequired<LoadModelPass>();
  }

public:
  bool runOnModule(llvm::Module &M) override final;

  static void writeModel(model::Binary &Model, llvm::Module &M);
};
