#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/PassManager.h"

#include "revng/TypeShrinking/RewritePlans.h"

namespace TypeShrinking {

class TypeShrinkingAnalysisPass
  : public llvm::AnalysisInfoMixin<TypeShrinkingAnalysisPass> {
  friend llvm::AnalysisInfoMixin<TypeShrinkingAnalysisPass>;

public:
  using Result = RewritePlans;

private:
  static llvm::AnalysisKey Key;

public:
  /// Plan rewrites using LLVM demanded bits and SCCP value ranges.
  /// Plan only reachable blocks, in reverse post-order, and leave the IR
  /// unchanged. Plan nothing in a function with an `invoke`.
  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &);
};

} // namespace TypeShrinking
