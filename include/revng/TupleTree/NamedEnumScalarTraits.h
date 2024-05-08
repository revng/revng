#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

template<typename E>
struct NamedEnumScalarTraits {
  template<typename T>
  static void enumeration(T &IO, E &V) {
    for (unsigned I = 0; I < E::Count; ++I) {
      auto Value = static_cast<E>(I);
      IO.enumCase(V, getName(Value).data(), Value);
    }
  }
};
