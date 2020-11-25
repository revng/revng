
//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include "revng-c/FilterForDecompilation/FilterForDecompilationPass.h"

static bool isIsolated(const llvm::Function &F) {
  return F.getMetadata("revng.func.entry");
}

bool FilterForDecompilationFunctionPass::runOnFunction(llvm::Function &F) {
  if (not isIsolated(F)) {
    F.deleteBody();
    return true;
  }

  return false;
}

char FilterForDecompilationFunctionPass::ID = 0;

bool FilterForDecompilationModulePass::runOnModule(llvm::Module &M) {

  bool Changed = false;
  for (llvm::Function &F : M) {
    if (not isIsolated(F)) {
      F.deleteBody();
      Changed = true;
    }
  }

  return Changed;
}

char FilterForDecompilationModulePass::ID = 0;

using llvm::RegisterPass;
using Pass = FilterForDecompilationModulePass;
static RegisterPass<Pass> X("filter-for-decompilation",
                            "Delete the body of all non-isolated functions");
