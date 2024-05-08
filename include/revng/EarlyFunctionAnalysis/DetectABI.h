#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/Pass.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"

namespace efa {

class DetectABIPass : public llvm::ModulePass {
public:
  static char ID;

public:
  DetectABIPass() : llvm::ModulePass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
    AU.addRequired<LoadModelWrapperPass>();
  }

  bool runOnModule(llvm::Module &M) override;
};

} // namespace efa
