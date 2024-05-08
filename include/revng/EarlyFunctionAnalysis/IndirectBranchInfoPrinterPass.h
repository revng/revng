#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/PassManager.h"

class IndirectBranchInfoPrinterPass
  : public llvm::PassInfoMixin<IndirectBranchInfoPrinterPass> {
  llvm::raw_ostream &OS;

public:
  IndirectBranchInfoPrinterPass(llvm::raw_ostream &OS) : OS(OS){};

  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);

private:
  void serialize(llvm::CallBase *Call);
};
