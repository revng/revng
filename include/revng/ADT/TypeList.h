#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <tuple>

#include "revng/ADT/Concepts.h"

template<typename... Ts>
struct TypeList {};

template<size_t I, typename... Ts>
struct std::tuple_element<I, TypeList<Ts...>>
  : public std::tuple_element<I, std::tuple<Ts...>> {};

template<typename... Ts>
struct std::tuple_size<TypeList<Ts...>>
  : public std::tuple_size<std::tuple<Ts...>> {};

/// Calls \ref Callable on each element of \ref TupleType. Each time the element
/// type and index will be provided as template parameters.
template<SpecializationOf<TypeList> TL, typename CallableType>
constexpr void forEach(CallableType &&Callable) {
  auto Runner = [&Callable]<size_t... I>(std::index_sequence<I...>) {
    (Callable.template operator()<std::tuple_element_t<I, TL>, I>(), ...);
  };
  Runner(std::make_index_sequence<std::tuple_size_v<TL>>{});
}

namespace detail {

template<typename... Args1, typename... Args2>
TypeList<Args1..., Args2...> concat(TypeList<Args1...>, TypeList<Args2...>);

} // namespace detail

/// Helper using that concatenates the types of two `TypeList`s together
template<SpecializationOf<TypeList> T1, SpecializationOf<TypeList> T2>
using concat = decltype(detail::concat(std::declval<T1>(), std::declval<T2>()));
