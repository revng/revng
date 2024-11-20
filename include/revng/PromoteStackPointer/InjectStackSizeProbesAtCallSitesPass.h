#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

/// Inject calls to `stack_size_at_call_site` to record stack size at call
/// sites
///
/// \note This pass injects loads to the stack pointer CSV, therefore it needs
///       to be run *before* `PromoteStackPointerPass`
struct InjectStackSizeProbesAtCallSitesPass : public llvm::ModulePass {
public:
  static char ID;

  InjectStackSizeProbesAtCallSitesPass() : llvm::ModulePass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool runOnModule(llvm::Module &M) override;
};
