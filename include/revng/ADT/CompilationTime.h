#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <optional>
#include <string_view>
#include <type_traits>

#include "llvm/ADT/StringRef.h"

#include "revng/ADT/Concepts.h"

namespace compile_time {

namespace detail {

template<typename T>
using RVHelper = decltype(std::declval<T>().template operator()<0>());

template<typename TemplatedCallableType, size_t... Indices>
  requires(std::is_same_v<RVHelper<TemplatedCallableType>, void>)
constexpr void
repeat(std::index_sequence<Indices...>, TemplatedCallableType &&Callable) {
  (Callable.template operator()<Indices>(), ...);
}

template<typename TemplatedCallableType, size_t... Indices>
  requires(!std::is_same_v<RVHelper<TemplatedCallableType>, void>)
constexpr auto
repeat(std::index_sequence<Indices...>, TemplatedCallableType &&Callable) {
  return std::tie(Callable.template operator()<Indices>()...);
}

template<typename TemplatedCallableType, size_t... Indices>
constexpr bool
repeatAnd(std::index_sequence<Indices...>, TemplatedCallableType &&Callable) {
  return (Callable.template operator()<Indices>() && ...);
}

template<typename TemplatedCallableType, size_t... Indices>
constexpr bool
repeatOr(std::index_sequence<Indices...>, TemplatedCallableType &&Callable) {
  return (Callable.template operator()<Indices>() || ...);
}

template<typename TemplatedCallableType, size_t... Indices>
constexpr size_t
count(std::index_sequence<Indices...>, TemplatedCallableType &&Callable) {
  return ((Callable.template operator()<Indices>() ? 1 : 0) + ...);
}

template<typename TemplatedCallableType, size_t... Indices>
constexpr std::optional<size_t>
select(std::index_sequence<Indices...> Is, TemplatedCallableType &&Callable) {
  if (count(Is, Callable) == 1)
    return ((Callable.template operator()<Indices>() ? Indices : 0) + ...);
  else
    return std::nullopt;
}

} // namespace detail

/// Calls \ref Callable \ref IterationCount times.
template<size_t IterationCount, typename CallableType>
constexpr auto repeat(CallableType &&Callable) {
  if constexpr (IterationCount > 0)
    return detail::repeat(std::make_index_sequence<IterationCount>(),
                          std::forward<CallableType>(Callable));
}

/// Calls \ref Callable \ref IterationCount times, while applying logical AND
/// operation to the return values.
template<size_t IterationCount, typename CallableType>
constexpr bool repeatAnd(CallableType &&Callable) {
  return detail::repeatAnd(std::make_index_sequence<IterationCount>(),
                           std::forward<CallableType>(Callable));
}

/// Calls \ref Callable \ref IterationCount times, while applying logical OR
/// operation to the return values.
template<size_t IterationCount, typename CallableType>
constexpr bool repeatOr(CallableType &&Callable) {
  return detail::repeatOr(std::make_index_sequence<IterationCount>(),
                          std::forward<CallableType>(Callable));
}

/// Calls \ref Callable \ref IterationCount times, makes sure at most one of
/// those invocations has a non-zero return value, then returns its index if
/// there is one, or `std::nullopt` if there's none.
template<size_t IterationCount, typename CallableType>
constexpr std::optional<size_t> select(CallableType &&Callable) {
  return detail::select(std::make_index_sequence<IterationCount>(),
                        std::forward<CallableType>(Callable));
}

namespace detail {

template<size_t N, size_t I = 0>
inline constexpr bool split(std::array<llvm::StringRef, N> &Result,
                            llvm::StringRef Separator,
                            llvm::StringRef Input) {
  size_t Position = Input.find(Separator);
  if constexpr (I < N - 1) {
    if (Position == llvm::StringRef::npos)
      return false;

    Result[I] = Input.substr(0, Position);
    return split<N, I + 1>(Result, Separator, Input.substr(Position + 1));
  } else {
    if (Position != llvm::StringRef::npos)
      return false;

    Result[I] = Input;
    return true;
  }
}

} // namespace detail

/// I'm forced to implement my own split because `llvm::StringRef`'s alternative
/// is not `constexpr`-compatible.
///
/// This also uses `llvm::StringRef` instead of `llvm::StringRef` because its
/// `find` member is constexpr - hence at least that member doesn't have to be
/// reimplemented
template<size_t N>
inline constexpr std::optional<std::array<llvm::StringRef, N>>
split(llvm::StringRef Separator, llvm::StringRef Input) {
  if (std::array<llvm::StringRef, N> Result;
      detail::split<N>(Result, Separator, Input))
    return Result;
  else
    return std::nullopt;
}

namespace detail {

template<typename>
struct ArrayTraits {};

template<typename T, size_t N>
struct ArrayTraits<T[N]> {
  using value_type = T;
  static constexpr size_t Size = N;
};

template<typename T, size_t N>
struct ArrayTraits<std::array<T, N>> {
  using value_type = T;
  static constexpr size_t Size = N;
};

} // namespace detail

/// Helper struct that reports the value_type and Size of an array at
/// compile-time
template<auto &T>
using ArrayTraits = detail::ArrayTraits<std::remove_cvref_t<decltype(T)>>;

} // namespace compile_time
