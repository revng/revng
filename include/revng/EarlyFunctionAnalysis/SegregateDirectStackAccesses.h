#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/PassManager.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"

class SegregateDirectStackAccessesPass
  : public llvm::PassInfoMixin<SegregateDirectStackAccessesPass> {
private:
  GeneratedCodeBasicInfo &GCBI;

public:
  SegregateDirectStackAccessesPass(GeneratedCodeBasicInfo &GCBI) : GCBI(GCBI) {}

  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};
