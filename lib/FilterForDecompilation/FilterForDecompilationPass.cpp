
//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include "revng/Model/LoadModelPass.h"

#include "revng-c/FilterForDecompilation/FilterForDecompilationPass.h"
#include "revng-c/IsolatedFunctions/IsolatedFunctions.h"

using FFDFP = FilterForDecompilationFunctionPass;

void FFDFP::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<LoadModelPass>();
}

bool FFDFP::runOnFunction(llvm::Function &F) {
  const model::Binary &Model = getAnalysis<LoadModelPass>().getReadOnlyModel();
  if (not hasIsolatedFunction(Model, F)) {
    F.deleteBody();
    return true;
  }

  return false;
}

char FFDFP::ID = 0;

using FFDMP = FilterForDecompilationModulePass;

void FFDMP::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<LoadModelPass>();
}

bool FFDMP::runOnModule(llvm::Module &M) {

  bool Changed = false;
  const model::Binary &Model = getAnalysis<LoadModelPass>().getReadOnlyModel();
  for (llvm::Function &F : M) {
    if (not hasIsolatedFunction(Model, F)) {
      F.deleteBody();
      Changed = true;
    }
  }

  return Changed;
}

char FFDMP::ID = 0;

using llvm::RegisterPass;
using Pass = FFDMP;
static RegisterPass<Pass> X("filter-for-decompilation",
                            "Delete the body of all non-isolated functions");
