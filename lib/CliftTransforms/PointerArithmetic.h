#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/Value.h"

#include "revng/Clift/Clift.h"
#include "revng/Support/Debug.h"

/// Represents a pointer-typed expression decomposed into a `BasePointer` and an
/// `Offset` expression. The offset is a constant `BaseOffset`, plus a linear
/// combination of strided terms, which capture array index patterns
struct PointerArithmetic {

  /// Represents an `Index` with both a variable and a constant component,
  /// e.g., `array[i+4]`
  struct Index {
    mlir::Value Variable;
    llvm::APInt Constant;
  };

  /// Represents a strided term in the `PointerArithmetic`
  struct StridedTerm {
    llvm::APInt Stride;
    Index Idx;

    StridedTerm(llvm::APInt Stride, Index Idx) :
      Stride(std::move(Stride)), Idx(std::move(Idx)) {}

    void dump() const debug_function;
  };

  /// Represents the offset with possible strided terms in the
  /// `PointerArithmetic`
  struct OffsetExpression {

    /// Represents the constant part of the `Offset`
    llvm::APInt BaseOffset;

    /// Holds the terms of the linear combination components of the `Offset`
    llvm::SmallVector<StridedTerm> LinearCombination;

    OffsetExpression(unsigned BitWidth);
    OffsetExpression(llvm::APInt Offset);

    void dump() const debug_function;
  };

  /// The base pointer the `PointerArithmetic` is expressed relative to
  /// `BasePointer` object, which is where the root of the computation lies
  mlir::Value BasePointer;

  /// The offset w.r.t. the `BasePointer`
  OffsetExpression Offset;

  /// We define a `PointerArithmetic` with an empty `BasePointer` a _numeric_
  bool isNumeric() const;

  /// Check if this is an address (has base pointer) arithmetic
  bool isAddress() const;

  /// Verify the invariants for the strides contained in the PointerArithmetic
  bool verify() const;

  /// Dump method
  void dump() const debug_function;
};

/// `PointerArithmetic` computation function entrypoint
std::optional<PointerArithmetic>
computePointerArithmetic(clift::ExpressionOpInterface PointerToReplace);
