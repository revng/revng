/// Promote CSVs in form of global variables to local variables.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"

#include "revng/EarlyFunctionAnalysis/PromoteGlobalToLocalVars.h"
#include "revng/Support/GlobalToLocalPromoter.h"
#include "revng/Support/IRBuilder.h"

using namespace llvm;

llvm::PreservedAnalyses
PromoteGlobalToLocalPass::run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM) {
  // Replace all the uses of each CSV, then load the CSV and store its value
  // onto the local variable. Each load is created after the replacement, so
  // that it reads the CSV and not the local variable that has just taken its
  // place.
  revng::IRBuilder Builder(F.getContext());
  Builder.SetInsertPointPastAllocas(&F);
  for (auto [CSV, Alloca] : promoteGlobalsToLocals(F, ShouldPromote))
    Builder.CreateStore(Builder.createLoad(CSV), Alloca);

  return PreservedAnalyses::none();
}
