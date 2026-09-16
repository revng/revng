#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueHandle.h"

/// When one value is written to two CSVs, make the second write read the first.
///
/// We go from:
///
///     store %v, @_rdx
///     store %v, @_cc_dst
///
/// To:
///
///     store %v, @_rdx
///     %0 = load @_rdx
///     store %0, @_cc_dst
///
/// This ensures the write on rdx does not seem dead.
class ChainCSVWritesPass : public llvm::PassInfoMixin<ChainCSVWritesPass> {

private:
  llvm::SmallVectorImpl<llvm::WeakVH> &Loads;

public:
  explicit ChainCSVWritesPass(llvm::SmallVectorImpl<llvm::WeakVH> &Loads) :
    Loads(Loads) {}

  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};
