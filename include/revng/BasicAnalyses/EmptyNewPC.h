#ifndef EMPTYNEWPC_H
#define EMPTYNEWPC_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include "llvm/Pass.h"

class EmptyNewPC : public llvm::ModulePass {
public:
  static char ID;

public:
  EmptyNewPC() : llvm::ModulePass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  bool runOnModule(llvm::Module &M) override;
};

#endif // EMPTYNEWPC_H
