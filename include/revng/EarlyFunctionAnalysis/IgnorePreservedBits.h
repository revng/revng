#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/PassManager.h"

/// Ignore spurious loads in RUA when doing `rdx = (rdx & 0xFF00 | somevalue)`
///
/// For example `sete %dl` lifts to:
///
///     %0 = load i64, ptr @_rdx
///     %1 = and i64 %0, u0xffffffffffffff00
///     %2 = or i64 %1, %condition
///     store i64 %2, ptr @_rdx
///
/// We want to ignore the load.
class IgnorePreservedBitsPass
  : public llvm::PassInfoMixin<IgnorePreservedBitsPass> {

public:
  IgnorePreservedBitsPass() = default;

  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};
