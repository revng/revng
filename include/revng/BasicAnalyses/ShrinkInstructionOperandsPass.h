#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/PassManager.h"

/// Transformation to shrink operand sizes where possible
///
/// This pass shrinks the operand of binary operators and comparison
/// instructions if they are zero/sign-extended immediately before and after the
/// instruction.
///
/// For instance:
///
///     %op1 = zext i32 0 to i64
///     %op2 = zext i32 0 to i64
///     %cmp = icmp ugt i64 %op1, %op2
///
/// Becomes:
///
///     %cmp = icmp ugt i32 0, 0
///
/// This enables other analyses (LazyValueInfo in particular) to obtain more
/// accurate results.
class ShrinkInstructionOperandsPass
  : public llvm::PassInfoMixin<ShrinkInstructionOperandsPass> {

public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};
