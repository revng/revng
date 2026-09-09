#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <functional>

#include "llvm/IR/PassManager.h"

namespace llvm {
class GlobalVariable;
} // namespace llvm

class PromoteGlobalToLocalPass
  : public llvm::PassInfoMixin<PromoteGlobalToLocalPass> {

public:
  /// Which CSVs to promote. An empty filter promotes all of them.
  using Filter = std::function<bool(const llvm::GlobalVariable &)>;

private:
  Filter ShouldPromote;

public:
  PromoteGlobalToLocalPass() = default;

  explicit PromoteGlobalToLocalPass(Filter ShouldPromote) :
    ShouldPromote(std::move(ShouldPromote)) {}

  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};
