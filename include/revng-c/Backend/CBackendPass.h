#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

struct BackendPass : public llvm::FunctionPass {
  static char ID;

  BackendPass();

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

private:
  std::unique_ptr<llvm::raw_ostream> DecompiledOStream;
};
