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

static_assert(is_specialization_v<std::vector<int>, std::vector>);
static_assert(is_specialization_v<std::pair<int, long>, std::pair>);

//
// HasTupleSize
//

template<typename T>
concept HasTupleSize = requires {
  std::tuple_size<T>::value;
};

static_assert(HasTupleSize<std::tuple<>>);
static_assert(!HasTupleSize<std::vector<int>>);
static_assert(!HasTupleSize<int>);
