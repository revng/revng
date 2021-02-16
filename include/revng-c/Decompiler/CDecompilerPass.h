#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <memory>

#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

struct CDecompilerPass : public llvm::FunctionPass {
  static char ID;

  CDecompilerPass();

  CDecompilerPass(std::unique_ptr<llvm::raw_ostream> Out);

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

private:
  std::unique_ptr<llvm::raw_ostream> Out;
};
