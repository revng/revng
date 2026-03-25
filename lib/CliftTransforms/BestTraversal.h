#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>
#include <optional>
#include <vector>

#include "llvm/ADT/SmallVector.h"

#include "revng/Support/Debug.h"

#include "PointerArithmetic.h"

/// `NestedArrayShape` represents the shape of a nested array element within a
/// type traversal, including its offset from the parent array element, the
/// number of elements, and the stride between consecutive elements
struct NestedArrayShape {

  // Represents the offset from the last _parent array_ element traversed to
  // reach this array
  int64_t OffsetFromParentArrayElement;
  uint64_t NumElements;
  uint64_t Stride;
};

/// A path of traversed `array`s is represented as a vector. These are kept
/// sorted with larger strides first, and we assume that there are no duplicated
/// strides
using ArrayPath = std::vector<NestedArrayShape>;

/// The `ArrayShape` struct is used to represent a generic array shape, where
/// the concrete accessed element is not taken into consideration
struct ArrayShape {
  uint64_t NumElements; ///< Size of the described array
  uint64_t Stride; ///< Stride of the described array

  /// Orders `ArrayShape`s by descending `Stride`, then ascending `NumElements`
  friend std::strong_ordering operator<=>(const ArrayShape &LHS,
                                          const ArrayShape &RHS) {
    // Descending Stride, then ascending NumElements
    if (auto Cmp = RHS.Stride <=> LHS.Stride; Cmp != 0)
      return Cmp;
    return LHS.NumElements <=> RHS.NumElements;
  }

  friend bool operator==(const ArrayShape &LHS,
                         const ArrayShape &RHS) = default;
};

/// This represents a type traversal starting from a fixed `BaseType`.
/// A Traversal represents a way to traverse `BaseType` accessing struct fields,
/// union fields and array elements, to reach a given fixed total offset from
/// the beginning of `BaseType`. The total offset is composed of two components:
///
/// * The `StartOffset`, which is the start offset of the innermost nested
/// target field that the traversal reaches
/// * The `LeftoverOffset`, which is the additional offset inside that target
/// field, not explicitly captured by the `Traversal`.
///
/// Note: The index of the traversal for an array is not specified.
///       In this sense, the traversal is "abstract".
/// Note: `StartOffset` behaves like if the first element of each
///       array is always traversed.
struct Traversal {

  /// The nested type, inside `BaseType`, where the `Traversal` lands
  mlir::Type TargetType;

  /// The start offset, from a fixed `BaseType`, which is "consumed" by this
  /// `Traversal`
  int64_t StartOffset;

  /// Additional offset inside the `TargetType`, not explicitly covered by the
  /// `Traversal`
  int64_t LeftoverOffset;

  /// The index (position in the fields array) of each traversed
  /// `union`/`struct` field. For `struct`s, this is the positional index of the
  /// field, not the byte offset. For unions, all fields start at offset 0 so
  /// this is also the positional index.
  std::vector<uint64_t> TraversedFields;

  /// Sorted vector of `ArrayShape` describing the array traversals, ordered in
  /// descending order by `Stride` (via `operator<` on `ArrayShape`). There can
  /// be consecutive `ArrayShape`s with the same `Stride`, to allow expressing
  /// e.g., `int array[1][1]`.
  llvm::SmallVector<ArrayShape> TraversedArrays;

  /// Construct a `Traversal` from its components, converting the `ArrayPath`
  /// to a sorted `SmallVector<ArrayShape>` internally
  Traversal(mlir::Type TargetType,
            int64_t StartOffset,
            int64_t LeftoverOffset,
            std::vector<uint64_t> TraversedFields,
            const ArrayPath &ArrayPath);

  /// Total depth described by this `Traversal` (fields + array elements)
  int64_t depth() const;

  /// Initial offset accessed by the `Traversal`
  int64_t begin() const;

  /// First out of bound offset accessed by the `Traversal`, considering the
  /// size of the `PointeeType`
  int64_t end() const;

  /// Debug `dump` method used to provide a textual representation on the logger
  /// of the `Traversal`
  void dump() const debug_function;
};

/// `TraversalInfo` describes a set of `Traversal`s and `ArrayPaths`, which we
/// associated to a `mlir::Type` (so we can cache them)
struct TraversalInfo {
  std::vector<Traversal> Traversals;
  std::vector<ArrayPath> ArrayPaths;
};

/// The map used to cache the `Traversal`s and `ArrayPath`s computed starting
/// from a `mlir::Type`
using TraversalInfoMap = llvm::DenseMap<mlir::Type, TraversalInfo>;

/// Helper function used to obtain the `BaseType` for the `Traversal` analysis.
/// Specifically, this is used to enable the `p[i]` rewrites. In situations
/// where we have a `pointer` to a `PrimitiveType`, we virtually wrap it in a
/// `array<N x T>`.
mlir::Type deriveBaseType(mlir::Value BasePointer);

/// Main entry point used to compute the `BestTraversal` from an `Expression`
/// and PointerArithmetic`.
std::optional<Traversal>
computeBestTraversal(mlir::clift::ExpressionOpInterface PointerToReplace,
                     const PointerArithmetic &Arithmetic,
                     TraversalInfoMap &Data);
