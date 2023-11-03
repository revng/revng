#pragma once
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/Concepts.h"
#include "revng/ADT/ConstexprString.h"

template<typename T>
using remove_constptr = std::remove_const_t<std::remove_pointer_t<T>>;

template<typename T>
inline constexpr bool isInteger() {
  return anyOf<T, int, uint8_t, uint16_t, uint32_t, uint64_t>();
}

template<typename T>
concept IntegerType = isInteger<T>();

template<typename T>
inline constexpr bool isList() {
  return std::is_pointer_v<T>
         && (isInteger<std::remove_pointer_t<T>>()
             || std::is_pointer_v<std::remove_pointer_t<T>>);
}

template<typename T>
using max_int = std::conditional_t<std::is_signed_v<T>, int64_t, uint64_t>;

template<ConstexprString Name>
inline constexpr bool isDestroy() {
  return std::string_view(Name).ends_with("_destroy");
}

inline constexpr auto PointerPrefix = "ptr_";
inline constexpr auto NullPointer = "ptr_0x0";
