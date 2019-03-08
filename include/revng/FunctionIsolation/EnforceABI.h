#ifndef ENFORCEABI_H
#define ENFORCEABI_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <memory>

// LLVM includes
#include "llvm/Pass.h"

// Local libraries includes
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/StackAnalysis/StackAnalysis.h"

class EnforceABI : public llvm::ModulePass {
public:
  static char ID;

public:
  EnforceABI() : ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<GeneratedCodeBasicInfo>();
    AU.setPreservesAll();
  }
};

#endif // ENFORCEABI_H
