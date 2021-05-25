#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/PassManager.h"

#include "revng/Support/revng.h"

class IndirectBranchInfoPrinterPass
  : public llvm::PassInfoMixin<IndirectBranchInfoPrinterPass> {
  llvm::raw_fd_ostream &OS;

public:
  IndirectBranchInfoPrinterPass(llvm::raw_fd_ostream &OS) : OS(OS){};

  llvm::PreservedAnalyses
  run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);

private:
  void serialize(llvm::CallBase *Call);
};
