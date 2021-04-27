#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <type_traits>

#include "llvm/ADT/STLExtras.h"

//
// is_integral
//
template<typename T>
concept Integral = std::is_integral_v<T>;

//
// is_specialization
//
template<typename Test, template<typename...> class Ref>
struct is_specialization : std::false_type {};

template<template<typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref> : std::true_type {};

template<typename Test, template<typename...> class Ref>
constexpr bool is_specialization_v = is_specialization<Test, Ref>::value;

//
// HasTupleSize
//

template<typename T>
concept HasTupleSize = requires {
  std::tuple_size<T>::value;
};

// clang-format off
template<typename T, std::size_t I>
concept IsTupleEnd
  = std::is_same_v<std::true_type,
                   std::integral_constant<bool, std::tuple_size_v<T> == I>>;

template<typename T, std::size_t I>
concept IsNotTupleEnd
  = std::is_same_v<std::true_type,
                   std::integral_constant<bool, std::tuple_size_v<T> != I>>;
// clang-format on

static_assert(HasTupleSize<std::tuple<>>);
static_assert(!HasTupleSize<std::vector<int>>);
static_assert(!HasTupleSize<int>);
