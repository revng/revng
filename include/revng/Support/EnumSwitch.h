#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <cstddef>
#include <type_traits>

#include "revng/ADT/Concepts.h"
#include "revng/Support/Debug.h"

namespace revng::detail {

template<typename T>
concept EnumWithCount = requires {
  requires std::is_enum<typename std::decay<T>::type>::value;
  { std::decay_t<T>::Count } -> std::convertible_to<size_t>;
};

} // namespace revng::detail

/// Calls the `operator()<Enumerator>()` of the callable `F` object passed
/// as the second parameter, where `Enumerator` corresponds to the runtime
/// argument passed as the first parameter (`Value`).
///
/// This can also be used as a simple way to instantiate a templated object
/// or function for every possible value of the specified enumeration.
///
/// \note: it requires the enumeration to contain the `Count` element to
/// mark the "one-after-the-last" enumerator.
///
/// \note: the behaviour is undefined if the enumeration is not continuous.
template<revng::detail::EnumWithCount Enum,
         size_t From = 0,
         size_t To = size_t(std::decay_t<Enum>::Count)>
constexpr inline auto enumSwitch(Enum Value, const auto &F) {
  using Decayed = typename std::decay<Enum>::type;

  constexpr Decayed Current = static_cast<Decayed>(From);
  if (Current == Value)
    return F.template operator()<Current>();
  else if constexpr (From + 1 < To)
    return enumSwitch<Enum, From + 1>(Value, F);
  else
    revng_abort("Unknown and/or unsupported enum value was encountered");
}

/// A specialized version of `enumSwitch` that allows skipping `SkippedCount`
/// first enumerators. The most typical use case it so avoid instantiating
/// templates for the first `Invalid = 0` enumerator.
template<size_t SkippedCount, revng::detail::EnumWithCount Enum>
constexpr inline auto skippingEnumSwitch(Enum Value, const auto &F) {
  return enumSwitch<Enum, SkippedCount>(Value, F);
}
