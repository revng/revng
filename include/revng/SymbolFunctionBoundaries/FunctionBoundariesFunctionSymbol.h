#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>

#include "llvm/Pass.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"

class FunctionBoundariesFunctionSymbol : public llvm::ModulePass {
public:
  static char ID;

public:
  FunctionBoundariesFunctionSymbol() : ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
    AU.setPreservesAll();
  }
};
