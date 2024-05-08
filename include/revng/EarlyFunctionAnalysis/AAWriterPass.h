#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/PassManager.h"

class AAWriterPass : public llvm::PassInfoMixin<AAWriterPass> {
  llvm::raw_ostream &OS;
  const bool StoresOnly;

public:
  AAWriterPass(llvm::raw_ostream &OS, bool StoresOnly = false) :
    OS(OS), StoresOnly(StoresOnly){};

  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};
