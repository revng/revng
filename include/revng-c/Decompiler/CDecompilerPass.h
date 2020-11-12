#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <memory>

#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"

#include "revng-c/PHIASAPAssignmentInfo/PHIASAPAssignmentInfo.h"
#include "revng-c/RestructureCFGPass/RestructureCFG.h"

struct CDecompilerPass : public llvm::FunctionPass {
  static char ID;

  CDecompilerPass();
  CDecompilerPass(std::unique_ptr<llvm::raw_ostream> Out);

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<RestructureCFG>();
    AU.addRequired<PHIASAPAssignmentInfo>();
    AU.setPreservesAll();
  }

private:
  std::unique_ptr<llvm::raw_ostream> Out;
  std::string SourceCode;
};
