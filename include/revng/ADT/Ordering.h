#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>
#include <concepts>
#include <set>

#include "llvm/ADT/StringRef.h"

/// Orders pointers by the object they point at, rather than by the pointer.
///
/// Keying a container on bare pointers makes its iteration order the order
/// the allocator handed the objects out, which is why it is not reproducible.
/// This makes the order a property of the pointees instead.
///
/// A null pointer compares less than any other one.
template<typename T>
struct PointeeLess {
  bool operator()(const T *LHS, const T *RHS) const {
    if (LHS == nullptr)
      return RHS != nullptr;

    if (RHS == nullptr)
      return false;

    return *LHS < *RHS;
  }
};

/// Anything carrying a `name()` that reads as a string
template<typename T>
concept Named = requires(const T &Value) {
  { Value.name() } -> std::convertible_to<llvm::StringRef>;
};

/// Orders pointers to named objects by name, rather than by the pointer.
///
/// The requirement sits on the call rather than on the template so that a
/// container of this can be declared while the pointee is still incomplete,
/// which is how `FunctionTags::TagsSet` uses it.
///
/// A null pointer compares less than any other one.
template<typename T>
struct ByName {
  bool operator()(const T *LHS, const T *RHS) const
    requires Named<T>
  {
    if (LHS == nullptr)
      return RHS != nullptr;

    if (RHS == nullptr)
      return false;

    return LHS->name() < RHS->name();
  }
};

/// A set of named objects, iterated by name
template<typename T>
using NamedObjectsSet = std::set<const T *, ByName<T>>;

/// Three-way comparison for `llvm::StringRef`, which has no `operator<=>`
inline std::strong_ordering compareStrings(llvm::StringRef LHS,
                                           llvm::StringRef RHS) {
  int Result = LHS.compare(RHS);
  if (Result < 0)
    return std::strong_ordering::less;

  if (Result > 0)
    return std::strong_ordering::greater;

  return std::strong_ordering::equal;
}
