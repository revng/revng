#pragma once
//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <tuple>

#include "revng/PipelineC/PipelineC.h"

using RPTypes = std::tuple<
#define TYPE(name) name,
#include "revng/PipelineC/Types.inc"
#undef TYPE
  void>;

// Given a tuple type TupleT, this will say if the type T is one of the types
// contained within the tuple type
template<typename TupleT, typename T, size_t I = 0>
inline constexpr bool isIn() {
  using ElementT = typename std::tuple_element<I, TupleT>::type;
  if constexpr (std::is_same_v<T, ElementT>
                || std::is_same_v<const T, ElementT>) {
    return true;
  } else {
    // Account for the fact that RPTypes ends with a dummy `void`
    if constexpr (I + 1 < std::tuple_size_v<TupleT> - 1)
      return isIn<TupleT, T, I + 1>();
    else
      return false;
  }
}

template<typename T>
inline constexpr bool isRPType() {
  return isIn<RPTypes, T>();
};

template<typename T>
concept RPType = isRPType<T>();
