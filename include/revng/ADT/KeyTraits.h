#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/Twine.h"

#include "revng/ADT/STLExtras.h"

//
// KeyTraits
//
using KeyInt = uint64_t;
using KeyIntVector = std::vector<KeyInt>;

template<typename T, typename = void>
struct KeyTraits {
  // static constexpr size_t IntsCount = ...;
  // using IntsArray = std::array<KeyInt, IntsCount>;
  //
  // static T fromInts(const IntsArray &KeyAsInts) {
  //   ...
  // }
  //
  // static IntsArray toInts(const T &I) {
  //   ...
  // }
  //
  // static std::string toString(const T &I) {
  //   ...
  // }
};

/// Trivial specialization for integral types
template<typename T>
struct KeyTraits<T, enable_if_is_integral_t<T>> {
  static constexpr size_t IntsCount = 1;
  using IntsArray = std::array<KeyInt, IntsCount>;

  static T fromInts(const IntsArray &KeyAsInts) { return KeyAsInts[0]; }

  static IntsArray toInts(const T &I) { return { static_cast<KeyInt>(I) }; }

  static std::string toString(const T &I) { return llvm::Twine(I).str(); }
};
