#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

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
