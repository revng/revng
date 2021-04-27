
//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/FilterForDecompilation/FilterForDecompilationPass.h"

using FFDFP = FilterForDecompilationFunctionPass;

void FFDFP::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<LoadModelWrapperPass>();
}

bool FFDFP::runOnFunction(llvm::Function &F) {
  auto FTags = FunctionTags::TagsSet::from(&F);
  if (not FTags.contains(FunctionTags::Lifted)) {
    F.deleteBody();
    return true;
  }

  return false;
}

char FFDFP::ID = 0;

using FFDMP = FilterForDecompilationModulePass;

void FFDMP::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<LoadModelWrapperPass>();
}

bool FFDMP::runOnModule(llvm::Module &M) {

  bool Changed = false;
  for (llvm::Function &F : M) {
    auto FTags = FunctionTags::TagsSet::from(&F);
    if (not FTags.contains(FunctionTags::Lifted)) {
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
