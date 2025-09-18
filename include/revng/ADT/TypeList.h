#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <tuple>

template<typename... Ts>
struct TypeList {};

template<size_t I, typename... Ts>
struct std::tuple_element<I, TypeList<Ts...>>
  : public std::tuple_element<I, std::tuple<Ts...>> {};

template<typename... Ts>
struct std::tuple_size<TypeList<Ts...>>
  : public std::tuple_size<std::tuple<Ts...>> {};
