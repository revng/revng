#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <optional>
#include <string_view>
#include <type_traits>

#include "revng/ADT/ConstexprString.h"

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
inline constexpr bool split(std::array<std::string_view, N> &Result,
                            std::string_view Separator,
                            std::string_view Input) {
  size_t Position = Input.find(Separator);
  if constexpr (I < N - 1) {
    if (Position == std::string_view::npos)
      return false;

    Result[I] = Input.substr(0, Position);
    return split<N, I + 1>(Result, Separator, Input.substr(Position + 1));
  } else {
    if (Position != std::string_view::npos)
      return false;

    Result[I] = Input;
    return true;
  }
}

} // namespace detail

/// I'm forced to implement my own split because `llvm::StringRef`'s alternative
/// is not `constexpr`-compatible.
///
/// This also uses `std::string_view` instead of `llvm::StringRef` because its
/// `find` member is constexpr - hence at least that member doesn't have to be
/// reimplemented
template<size_t N>
inline constexpr std::optional<std::array<std::string_view, N>>
split(std::string_view Separator, std::string_view Input) {
  if (std::array<std::string_view, N> Result;
      detail::split<N>(Result, Separator, Input))
    return Result;
  else
    return std::nullopt;
}

/// Concatenates a bunch of string-like things at compilation time and returns
/// the result (with a /param Separator inserted in-between) as an array.
template<ConstexprString Separator, ConstexprString... Strings>
inline constexpr auto concatenateWithSeparator() {
  std::array<char,
             (std::size(Strings) + ...)
               + std::size(Separator) * (sizeof...(Strings) - 1)>
    Result;
  std::size_t CurrentIndex = 0;

  auto Impl = [&](auto const &String) {
    std::ranges::copy(String, Result.begin() + CurrentIndex);
    CurrentIndex += String.size();
    if (CurrentIndex < Result.size()) {
      std::ranges::copy(Separator, Result.begin() + CurrentIndex);
      CurrentIndex += Separator.size();
    }
  };

  (Impl(Strings), ...);

  return Result;
}
template<typename... StringLikeTypes>
inline constexpr auto concatenate(StringLikeTypes &&...Strs) {
  return concatenateWithSeparator("", std::forward<StringLikeTypes>(Strs)...);
}

} // namespace compile_time
