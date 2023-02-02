#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/KeyedObjectContainer.h"
#include "revng/TupleTree/TupleLikeTraits.h"

// clang-format off
template<typename T>
concept TupleTreeCompatible = (KeyedObjectContainer<T>
                               or UpcastablePointerLike<T>
                               or TupleLike<T>);

template<typename T>
concept NotTupleTreeCompatible = not TupleTreeCompatible<T>;
// clang-format on

template<typename T>
concept Verifiable = requires(const T &TT) {
  { TT.verify() } -> std::same_as<bool>;
};

template<typename T>
concept TupleTreeCompatibleAndVerifiable =
  (TupleTreeCompatible<T> and Verifiable<T>);
