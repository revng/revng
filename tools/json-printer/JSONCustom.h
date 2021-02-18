#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <fstream>

#include "llvm/Pass.h"

class JsonCustom : public llvm::ModulePass {
public:
  static char ID;

public:
  JsonCustom() : llvm::ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};
