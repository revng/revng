//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef REVNGC_ENFORCECFGCOMBINGPASS_H
#define REVNGC_ENFORCECFGCOMBINGPASS_H

// LLVM includes
#include <llvm/Pass.h>

// revng-c includes
#include "revng-c/RestructureCFGPass/RestructureCFG.h"

struct EnforceCFGCombingPass : public llvm::FunctionPass {
  static char ID;

  EnforceCFGCombingPass() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<RestructureCFG>();
  }

};

#endif // REVNGC_ENFORCECFGCOMBINGPASS_H
