
//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/FilterForDecompilation/FilterForDecompilationPass.h"

using FFDFP = FilterForDecompilationFunctionPass;

void FFDFP::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<LoadModelWrapperPass>();
}

/// \brief Drop the body of all non-lifted functions, and add `optnone` and
/// `noinline` attributes so that they can be eliminated by DCE.
static bool filterFunction(llvm::Function &F) {
  auto FTags = FunctionTags::TagsSet::from(&F);
  if (not FTags.contains(FunctionTags::Isolated)) {
    auto Attributes = F.getAttributes();
    F.deleteBody();

    // deleteBody() also removes all metadata, including tags, and attributes.
    // Since we still want tags, we have to add them again. A better way to do
    // this would be to remove all metadata except for `!revng.tags`.
    F.setAttributes(Attributes);
    F.addFnAttr(llvm::Attribute::AttrKind::OptimizeNone);
    F.addFnAttr(llvm::Attribute::AttrKind::NoInline);
    FTags.addTo(&F);

    return true;
  }

  return false;
}

bool FFDFP::runOnFunction(llvm::Function &F) {
  return filterFunction(F);
}

char FFDFP::ID = 0;

using FFDMP = FilterForDecompilationModulePass;

void FFDMP::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<LoadModelWrapperPass>();
}

bool FFDMP::runOnModule(llvm::Module &M) {

  bool Changed = false;
  for (llvm::Function &F : M)
    Changed = filterFunction(F);

  return Changed;
}

char FFDMP::ID = 0;

using llvm::RegisterPass;
using Pass = FFDMP;
static RegisterPass<Pass> X("filter-for-decompilation",
                            "Delete the body of all non-isolated functions");
