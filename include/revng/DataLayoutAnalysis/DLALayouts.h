#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>
#include <compare>
#include <cstdint>
#include <map>
#include <numeric>
#include <optional>
#include <set>

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Support/Debug.h"

namespace llvm {
class Value;
} // end namespace llvm

namespace dla {

/// A representation of a pointer to a type.
class LayoutTypePtr {
  const llvm::Value *V;
  unsigned FieldIdx;

public:
  static constexpr unsigned fieldNumNone = std::numeric_limits<unsigned>::max();

  explicit LayoutTypePtr(const llvm::Value *Val, unsigned Idx = fieldNumNone) :
    V(Val), FieldIdx(Idx) {}

  LayoutTypePtr() = default;
  ~LayoutTypePtr() = default;
  LayoutTypePtr(const LayoutTypePtr &) = default;
  LayoutTypePtr(LayoutTypePtr &&) = default;
  LayoutTypePtr &operator=(const LayoutTypePtr &) = default;
  LayoutTypePtr &operator=(LayoutTypePtr &&) = default;

  std::strong_ordering operator<=>(const LayoutTypePtr &Other) const = default;

  unsigned fieldNum() const { return FieldIdx; }

  void print(llvm::raw_ostream &Out) const;
  std::string toString() const debug_function;

  const llvm::Value &getValue() const { return *V; }

  bool isEmpty() const { return (V == nullptr); }
}; // end class LayoutTypePtr

using LayoutTypePtrVect = std::vector<LayoutTypePtr>;

} // end namespace dla
