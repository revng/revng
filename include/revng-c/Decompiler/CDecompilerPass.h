#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <memory>

#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

struct CDecompilerPass : public llvm::FunctionPass {
  static char ID;

  CDecompilerPass();

  CDecompilerPass(std::unique_ptr<llvm::raw_ostream> Out);

  bool doInitialization(llvm::Module &M) override;

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

private:
  // This is a unique pointer because we need runtime polymorphism, so that we
  // can either write to file or to a string.
  std::unique_ptr<llvm::raw_ostream> Out;
};
