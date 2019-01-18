//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

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
