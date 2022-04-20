#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

struct BackendPass : public llvm::FunctionPass {
  static char ID;

  BackendPass();

  BackendPass(std::unique_ptr<llvm::raw_ostream> Out);

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

private:
  // This is a unique pointer because we need runtime polymorphism, so that we
  // can either write to file or to a string.
  std::unique_ptr<llvm::raw_ostream> Out;
};
