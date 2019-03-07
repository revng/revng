#ifndef REVNGC_REMOVE_UNEXPECTED_PC_REMOVEUNEXPECTEDPC_H
#define REVNGC_REMOVE_UNEXPECTED_PC_REMOVEUNEXPECTEDPC_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <fstream>

// LLVM includes
#include "llvm/Pass.h"

class RemoveUnexpectedPC : public llvm::FunctionPass {
public:
  static char ID;

public:
  RemoveUnexpectedPC() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};

#endif // REVNGC_REMOVE_UNEXPECTED_PC_REMOVEUNEXPECTEDPC_H
