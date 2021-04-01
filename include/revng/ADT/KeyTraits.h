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

template<typename T>
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
};

/// Trivial specialization for integral types
template<Integral T>
struct KeyTraits<T> {
  static constexpr size_t IntsCount = 1;
  using IntsArray = std::array<KeyInt, IntsCount>;

  static T fromInts(const IntsArray &KeyAsInts) { return KeyAsInts[0]; }

  static IntsArray toInts(const T &I) { return { static_cast<KeyInt>(I) }; }
};

template<typename T>
concept Enum = std::is_enum_v<T>;

/// Trivial specialization for integral types
template<Enum T>
struct KeyTraits<T> {
  static constexpr size_t IntsCount = 1;
  using IntsArray = std::array<KeyInt, IntsCount>;

  static T fromInts(const IntsArray &KeyAsInts) {
    return static_cast<T>(KeyAsInts[0]);
  }

  static IntsArray toInts(const T &I) { return { static_cast<KeyInt>(I) }; }
};

//
// Derive KeyTraits from tuple-like of objects featuring KeyTraits
//
template<HasTupleSize T>
struct KeyTraits<T> {
private:
  template<size_t I = 0>
  static constexpr size_t computeIntsCount() {
    if constexpr (I != std::tuple_size_v<T>) {
      auto Result = KeyTraits<std::tuple_element_t<I, T>>::IntsCount;
      return Result + computeIntsCount<I + 1>();
    } else {
      return 0;
    }
  }

public:
  static constexpr size_t IntsCount = KeyTraits::computeIntsCount();
  using IntsArray = std::array<KeyInt, IntsCount>;

  template<size_t I = 0, size_t First = 0>
  constexpr static void populateKeyArray(IntsArray &Result, const T &Object) {
    if constexpr (I != std::tuple_size_v<T>) {
      using InnerKeyTraits = KeyTraits<std::tuple_element_t<I, T>>;
      constexpr auto Size = InnerKeyTraits::IntsCount;

      const auto &TupleEntry = InnerKeyTraits::toInts(get<I>(Object));
      for (size_t J = 0; J < Size; ++J)
        Result[First + J] = TupleEntry[J];

      KeyTraits::populateKeyArray<I + 1, First + Size>(Result, Object);
    }
  }

  static IntsArray toInts(const T &Object) {
    IntsArray Result{};
    KeyTraits::populateKeyArray(Result, Object);
    return Result;
  }

  template<size_t I = 0, size_t First = 0>
  constexpr static void setFields(T &Object, const IntsArray &Key) {
    if constexpr (I != std::tuple_size_v<T>) {
      using KeyTraits = KeyTraits<std::tuple_element_t<I, T>>;

      // Populate partial key
      constexpr auto Size = KeyTraits::IntsCount;
      std::array<KeyInt, Size> PartialKey;
      for (size_t J = 0; J < Size; ++J)
        PartialKey[J] = Key[First + J];

      // Create and fill the fields
      get<I>(Object) = KeyTraits::fromInts(PartialKey);

      // Recur
      setFields<I + 1, First + Size>(Object, Key);
    }
  }

  static T fromInts(const IntsArray &Key) {
    T Result;
    setFields(Result, Key);
    return Result;
  }
};
