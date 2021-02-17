#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <type_traits>

#include "llvm/ADT/STLExtras.h"

#include "revng/ADT/KeyedObjectTraits.h"

//
// is_integral
//
template<typename T, typename K = void>
using enable_if_is_integral_t = std::enable_if_t<std::is_integral_v<T>, K>;

//
// enable_if_type
//
template<class T, class R = void>
struct enable_if_type {
  using type = R;
};

//
// is_specialization
//
template<typename Test, template<typename...> class Ref>
struct is_specialization : std::false_type {};

template<template<typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref> : std::true_type {};

template<typename Test, template<typename...> class Ref>
constexpr bool is_specialization_v = is_specialization<Test, Ref>::value;
