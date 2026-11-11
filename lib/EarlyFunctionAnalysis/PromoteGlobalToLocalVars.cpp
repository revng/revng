/// Promote CSVs in form of global variables to local variables.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include "revng/EarlyFunctionAnalysis/PromoteGlobalToLocalVars.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Support/GlobalToLocalPromoter.h"
#include "revng/Support/IRBuilder.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/OpaqueRegisterUser.h"

using namespace llvm;

llvm::PreservedAnalyses
PromoteGlobalToLocalPass::run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM) {

  revng::IRBuilder Builder(F.getContext());
  Builder.SetInsertPointPastAllocas(&F);

  // A constant global is never promotable. Promotion exists to turn mutable
  // state into something `mem2reg` can lift into SSA values, and a constant has
  // no such state. More importantly, the analyses read some constant globals
  // back by identity rather than by value: the `indirect_branch_info` marker
  // carries the caller's block ID and the called symbol's name as pointers to
  // constant string globals, and `milkInfo` decodes them with
  // `extractFromConstantStringPtr`, which only works if the argument still is
  // the global. Promote one of those and the argument becomes an alloca, the
  // block ID decodes to nothing, and looking it up in the CFG fails.
  auto IsPromotable = [this](const GlobalVariable &GV) {
    if (GV.isConstant())
      return false;

    return not ShouldPromote or ShouldPromote(GV);
  };

  // Create an equivalent local variable, replace all the uses of the CSV.
  GlobalToLocalPromoter Promoter(IsPromotable, F);
  for (GlobalVariable *CSV : Promoter.globals()) {
    auto *CSVTy = CSV->getValueType();
    auto *Alloca = Builder.CreateAlloca(CSVTy, nullptr, CSV->getName());
    Promoter.replaceWithAlloca(CSV, Alloca);
    // Load all the CSVs and store their value onto the local variables. These
    // loads are created after the replacement, so that they read the CSVs and
    // not the local variables that have just taken their place.
    Builder.CreateStore(Builder.createLoad(CSV), Alloca);
    // Reset insert point after the newly created alloca, ready for the next.
    Builder.SetInsertPoint(Alloca->getNextNode());
  }
  return PreservedAnalyses::none();
}
