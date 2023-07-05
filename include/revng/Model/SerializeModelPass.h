#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

#include "revng/Model/Binary.h"
#include "revng/Model/LoadModelPass.h"

void writeModel(const model::Binary &Model, llvm::Module &M);

class SerializeModelWrapperPass : public llvm::ModulePass {
public:
  static char ID;

public:
  SerializeModelWrapperPass() : llvm::ModulePass(ID) {}
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override final {
    AU.setPreservesAll();
    AU.addUsedIfAvailable<LoadModelWrapperPass>();
  }

public:
  bool runOnModule(llvm::Module &M) override final;
};

class SerializeModelPass : public llvm::PassInfoMixin<SerializeModelPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
};
