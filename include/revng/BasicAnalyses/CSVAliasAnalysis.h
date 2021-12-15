#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Support/revng.h"

class CSVAliasAnalysisPass : public llvm::PassInfoMixin<CSVAliasAnalysisPass> {

public:
  llvm::PreservedAnalyses
  run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);
};
