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
  /// Which CSVs to promote. An empty predicate promotes all of them.
  using SelectToPromote = std::function<bool(const llvm::GlobalVariable &)>;

private:
  SelectToPromote ShouldPromote;

public:
  PromoteGlobalToLocalPass() = default;

  explicit PromoteGlobalToLocalPass(SelectToPromote ShouldPromote) :
    ShouldPromote(std::move(ShouldPromote)) {}

  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};
