#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Model/Binary.h"
#include "revng/Model/LoadModelPass.h"

class CollectFunctionsFromUnusedAddressesWrapperPass : public llvm::ModulePass {
public:
  static char ID;

public:
  CollectFunctionsFromUnusedAddressesWrapperPass() : llvm::ModulePass(ID) {}
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override final {
    AU.setPreservesAll();
    AU.addRequired<LoadModelWrapperPass>();
    AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
  }

public:
  bool runOnModule(llvm::Module &M) override final;
};

class CollectFunctionsFromUnusedAddressesPass
  : public llvm::PassInfoMixin<CollectFunctionsFromUnusedAddressesPass> {
public:
  llvm::PreservedAnalyses
  run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);
};
