#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <tuple>

#include "llvm/ADT/StringRef.h"

//
// FOR_EACH macro implementation
//
#define GET_MACRO(_0,   \
                  _1,   \
                  _2,   \
                  _3,   \
                  _4,   \
                  _5,   \
                  _6,   \
                  _7,   \
                  _8,   \
                  _9,   \
                  _10,  \
                  _11,  \
                  _12,  \
                  _13,  \
                  _14,  \
                  _15,  \
                  _16,  \
                  NAME, \
                  ...)  \
  NAME
#define NUMARGS(...)     \
  GET_MACRO(_0,          \
            __VA_ARGS__, \
            16,          \
            15,          \
            14,          \
            13,          \
            12,          \
            11,          \
            10,          \
            9,           \
            8,           \
            7,           \
            6,           \
            5,           \
            4,           \
            3,           \
            2,           \
            1)

#define FE_0(ACTION, TOTAL, ARG)

#define FE_1(ACTION, TOTAL, ARG, X) ACTION(ARG, (TOTAL) -0, X)

#define FE_2(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -1, X)             \
  FE_1(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_3(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -2, X)             \
  FE_2(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_4(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -3, X)             \
  FE_3(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_5(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -4, X)             \
  FE_4(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_6(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -5, X)             \
  FE_5(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_7(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -6, X)             \
  FE_6(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_8(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -7, X)             \
  FE_7(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_9(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -8, X)             \
  FE_8(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_10(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -9, X)              \
  FE_9(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_11(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -10, X)             \
  FE_10(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_12(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -11, X)             \
  FE_11(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_13(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -12, X)             \
  FE_12(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_14(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -13, X)             \
  FE_13(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_15(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -14, X)             \
  FE_14(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_16(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -15, X)             \
  FE_15(ACTION, TOTAL, ARG, __VA_ARGS__)

/// Calls ACTION(ARG, INDEX, VA_ARG) for each VA_ARG in ...
#define FOR_EACH(ACTION, ARG, ...) \
  GET_MACRO(_0,                    \
            __VA_ARGS__,           \
            FE_16,                 \
            FE_15,                 \
            FE_14,                 \
            FE_13,                 \
            FE_12,                 \
            FE_11,                 \
            FE_10,                 \
            FE_9,                  \
            FE_8,                  \
            FE_7,                  \
            FE_6,                  \
            FE_5,                  \
            FE_4,                  \
            FE_3,                  \
            FE_2,                  \
            FE_1,                  \
            FE_0)                  \
  (ACTION, (NUMARGS(__VA_ARGS__) - 1), ARG, __VA_ARGS__)

//
// Macros to transform struct in tuple-like
//

#define TUPLE_TYPES(class, index, field) , decltype(class ::field)

#define TUPLE_FIELD_NAME(class, index, field) #field,

#define ENUM_ENTRY(class, index, field) field = index,

template<typename ToSkip, typename... A>
using skip_first_tuple = std::tuple<A...>;

#define INTROSPECTION_1(classname, ...)                                    \
  template<>                                                               \
  struct TupleLikeTraits<classname> {                                      \
    static constexpr const char *Name = #classname;                        \
    static constexpr const char *FullName = #classname;                    \
                                                                           \
    using tuple = skip_first_tuple<                                        \
      void FOR_EACH(TUPLE_TYPES, classname, __VA_ARGS__)>;                 \
                                                                           \
    static constexpr std::array<llvm::StringRef, std::tuple_size_v<tuple>> \
      FieldNames = { FOR_EACH(TUPLE_FIELD_NAME, classname, __VA_ARGS__) }; \
                                                                           \
    enum class Fields { FOR_EACH(ENUM_ENTRY, classname, __VA_ARGS__) };    \
  };

#define GET_IMPLEMENTATIONS(class, index, field) \
  else if constexpr (I == index) return x.field;

#define INTROSPECTION_2(class, ...)                   \
  template<int I>                                     \
  auto &get(class &&x) {                              \
    if constexpr (false)                              \
      return NULL;                                    \
    FOR_EACH(GET_IMPLEMENTATIONS, class, __VA_ARGS__) \
  }                                                   \
                                                      \
  template<int I>                                     \
  const auto &get(const class &x) {                   \
    if constexpr (false)                              \
      return NULL;                                    \
    FOR_EACH(GET_IMPLEMENTATIONS, class, __VA_ARGS__) \
  }                                                   \
                                                      \
  template<int I>                                     \
  auto &get(class &x) {                               \
    if constexpr (false)                              \
      return NULL;                                    \
    FOR_EACH(GET_IMPLEMENTATIONS, class, __VA_ARGS__) \
  }

#define INTROSPECTION(class, ...)     \
  INTROSPECTION_1(class, __VA_ARGS__) \
  INTROSPECTION_2(class, __VA_ARGS__)

#define INTROSPECTION_NS(ns, class, ...)  \
  INTROSPECTION_1(ns::class, __VA_ARGS__) \
  namespace ns {                          \
  INTROSPECTION_2(class, __VA_ARGS__)     \
  }
