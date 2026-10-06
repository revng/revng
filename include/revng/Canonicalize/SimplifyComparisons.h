#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/PassManager.h"

namespace revng {

/// Expose packed flag predicates and recover signed comparisons of narrow
/// values.
class SimplifyComparisonsPass
  : public llvm::PassInfoMixin<SimplifyComparisonsPass> {
public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};

} // namespace revng
