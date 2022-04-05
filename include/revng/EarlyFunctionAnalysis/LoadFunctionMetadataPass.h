#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

class LoadFunctionMetadataPass : public llvm::ModulePass {
public:
  const efa::FunctionMetadata &get(llvm::Function *F);

private:
  std::map<MetaAddress, efa::FunctionMetadata> EntryToMetadata;

public:
  static char ID;

public:
  LoadFunctionMetadataPass() : llvm::ModulePass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  bool runOnModule(llvm::Module &M) override;
};
