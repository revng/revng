#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/KeyTraits.h"
#include "revng/ADT/STLExtras.h"

template<typename T, typename = void>
struct KeyedObjectTraits {
  // static * key(const T &);
  // static T fromKey(* Key);
};

/// Inherit if T is the key of itself
template<typename T>
struct IdentityKeyedObjectTraits {
  static T key(const T &Obj) { return Obj; }

  static T fromKey(T Obj) { return Obj; }
};

/// Trivial specialization for integral types
template<Integral T>
struct KeyedObjectTraits<T> : public IdentityKeyedObjectTraits<T> {};
