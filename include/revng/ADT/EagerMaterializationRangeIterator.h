#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <compare>
#include <cstddef>
#include <iterator>
#include <optional>
#include <type_traits>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include "revng/Support/Debug.h"
#include "revng/Support/Generator.h"
#include "revng/Support/IRHelpers.h"

// Define a custom concept for restricting the T type
template<typename T>
concept TriviallyCopyable = std::is_trivially_copyable_v<T>
                            and std::is_copy_constructible_v<T>
                            and std::is_copy_assignable_v<T>;

// Iterator class that eagerly materializes a whole range of elements, holding a
// copy of it internally.
//
// We restrict the `T` type so that is a trivially copyable object.
template<TriviallyCopyable T>
class EagerMaterializationRangeIterator {
public:
  using difference_type = ssize_t;
  using value_type = T;
  using pointer = T *;
  using reference = T &;
  using iterator_category = typename std::bidirectional_iterator_tag;

private:
  llvm::SmallVector<T> MaterializedRange;
  size_t Index;

public:
  // Constructor building an empty materialized range.
  // An iterator holding this value compares equal to all other end iterators.
  EagerMaterializationRangeIterator() : MaterializedRange(), Index(0) {}

  EagerMaterializationRangeIterator(const llvm::SmallVector<T> &R) :
    MaterializedRange(R), Index(0) {}

  EagerMaterializationRangeIterator(llvm::SmallVector<T> &&R) :
    MaterializedRange(std::move(R)), Index(0) {}

public:
  // We assume the range reached the end when the `Index` points to the end
  bool isEnd() const { return Index == MaterializedRange.size(); }

  // static `end` method to quickly construct a
  // `EagerMaterializationRangeIterator` in the end state
  static EagerMaterializationRangeIterator end() {
    return EagerMaterializationRangeIterator();
  }

public:
  bool operator==(const EagerMaterializationRangeIterator &Other) const {
    if (isEnd() == Other.isEnd())
      return true;

    size_t AfterIndex = MaterializedRange.size() - Index;
    size_t AfterOtherIndex = Other.MaterializedRange.size() - Other.Index;
    if (AfterIndex != AfterOtherIndex)
      return false;

    auto It = MaterializedRange.begin() + Index;
    auto OtherIt = Other.MaterializedRange.begin() + Other.Index;

    for (const auto [L, R] :
         llvm::zip_equal(llvm::ArrayRef(It, AfterIndex),
                         llvm::ArrayRef(OtherIt, AfterOtherIndex))) {
      if (L != R)
        return false;
    }
    return true;
  }

  EagerMaterializationRangeIterator &operator++() {
    ++Index;
    return *this;
  }

  EagerMaterializationRangeIterator operator++(int) {
    EagerMaterializationRangeIterator BeforeIncrement(*this);
    ++*this;
    return BeforeIncrement;
  }

  EagerMaterializationRangeIterator &operator--() {
    --Index;
    return *this;
  }

  EagerMaterializationRangeIterator operator--(int) {
    EagerMaterializationRangeIterator BeforeDecrement(*this);
    --*this;
    return BeforeDecrement;
  }

  T operator*() const { return MaterializedRange[Index]; }
};
