#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/Concepts.h"
#include "revng/ADT/STLExtras.h"

template<typename T>
struct KeyedObjectTraits {
  // static * key(const T &);
  // static T fromKey(* Key);
};

// clang-format off
template<typename T>
concept HasKeyObjectTraits = requires(T A) {
  { KeyedObjectTraits<T>::key(A) };
  { KeyedObjectTraits<T>::fromKey(KeyedObjectTraits<T>::key(A)) } -> same_as<T>;
};
// clang-format on

/// Inherit if T is the key of itself
template<typename T>
struct IdentityKeyedObjectTraits {
  static T key(const T &Obj) { return Obj; }

  static T fromKey(T Obj) { return Obj; }
};

/// Trivial specializations
template<Integral T>
struct KeyedObjectTraits<T> : public IdentityKeyedObjectTraits<T> {};

template<>
struct KeyedObjectTraits<std::string>
  : public IdentityKeyedObjectTraits<std::string> {};

static_assert(Integral<int>);
static_assert(HasKeyObjectTraits<int>);
