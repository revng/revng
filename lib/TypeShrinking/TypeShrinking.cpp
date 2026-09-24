/// Shrink integer computations by applying the plans of
/// \ref TypeShrinking::TypeShrinkingAnalysisPass.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/TypeShrinking/TypeShrinking.h"
#include "revng/TypeShrinking/TypeShrinkingAnalysis.h"

using namespace llvm;

char TypeShrinking::TypeShrinkingWrapperPass::ID = 0;
using Register = RegisterPass<TypeShrinking::TypeShrinkingWrapperPass>;
static Register
  X("type-shrinking", "Shrink integer computations", false, false);

namespace TypeShrinking {

bool TypeShrinkingWrapperPass::runOnFunction(Function &F) {
  FunctionAnalysisManager FAM;
  return TypeShrinkingAnalysisPass().run(F, FAM).apply();
}

PreservedAnalyses TypeShrinkingPass::run(Function &F,
                                         FunctionAnalysisManager &FAM) {
  bool Changed = FAM.getResult<TypeShrinkingAnalysisPass>(F).apply();
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

} // namespace TypeShrinking
