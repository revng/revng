#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/ADT/BitVector.h"
#include "llvm/IR/Function.h"

namespace DetectUninlinableHelpers {

/// \return `true` if `Pointer` addresses a `GlobalVariable` marked as
///         `constant`.
bool isPointerToConstantGlobal(const llvm::Value *Pointer);

/// Compute the set of *critical formal arguments* of `Helper`.
///
/// An operand of an instruction is *critical* when one of the following holds:
/// - it is the condition of a `switch` instruction;
/// - it is one of the index operands of a `getelementptr` instruction.
///
/// A *critical argument* of `Helper` is a formal parameter that flows into a
/// critical operand. The function performs a backward dataflow walk from every
/// critical operand and classifies as critical every formal parameter reached
/// during the walk.
///
/// \return One of three possible values:
/// - `std::nullopt`: the helper cannot be inlined at any call site (the
///   backward walk reaches a `load` from runtime memory).
/// - empty `BitVector`: the helper can always be inlined.
/// - non-empty `BitVector`: the set of formal-parameter indices that must be
///   `isa<Constant>` at the call site for the helper to be inlinable.
std::optional<llvm::BitVector>
computeCriticalArgumentsFor(const llvm::Function &Helper);

} // namespace DetectUninlinableHelpers
