#ifndef ISOLATEFUNCTIONS_H
#define ISOLATEFUNCTIONS_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <memory>

// LLVM includes
#include "llvm/Pass.h"

// Local libraries includes
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"

class IsolateFunctions : public llvm::ModulePass {
public:
  static char ID;

public:
  IsolateFunctions() : ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<GeneratedCodeBasicInfo>();
  }
};

#endif // ISOLATEFUNCTIONS_H
