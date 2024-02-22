#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>

#include "llvm/Pass.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"

class EnforceABI : public pipeline::FunctionPassImpl {
public:
  using pipeline::FunctionPassImpl::FunctionPassImpl;
  bool prologue(llvm::Module &M, const model::Binary &Model) override;
  bool runOnFunction(llvm::Function &Function,
                     const model::Function &ModelFunction) override {
    return false;
  }

  static void getAnalysisUsage(llvm::AnalysisUsage &AU);
};
