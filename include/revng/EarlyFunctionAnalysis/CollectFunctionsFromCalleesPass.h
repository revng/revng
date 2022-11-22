#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Model/Binary.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"

class CollectFunctionsFromCalleesWrapperPass : public llvm::ModulePass {
public:
  static char ID;

public:
  CollectFunctionsFromCalleesWrapperPass() : llvm::ModulePass(ID) {}
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override final;

public:
  bool runOnModule(llvm::Module &M) override final;
};

class CollectFunctionsFromCalleesPass
  : public llvm::PassInfoMixin<CollectFunctionsFromCalleesPass> {
public:
  llvm::PreservedAnalyses
  run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);
};
