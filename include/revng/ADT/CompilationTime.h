#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <optional>
#include <string_view>
#include <type_traits>

namespace compile_time {

namespace detail {

template<typename TemplatedCallableType, std::size_t... Indices>
constexpr void
repeat(std::index_sequence<Indices...>, TemplatedCallableType &&Callable) {
  (Callable.template operator()<Indices>(), ...);
}

template<typename TemplatedCallableType, std::size_t... Indices>
constexpr bool
repeatAnd(std::index_sequence<Indices...>, TemplatedCallableType &&Callable) {
  return (Callable.template operator()<Indices>() && ...);
}

template<typename TemplatedCallableType, std::size_t... Indices>
constexpr bool
repeatOr(std::index_sequence<Indices...>, TemplatedCallableType &&Callable) {
  return (Callable.template operator()<Indices>() || ...);
}

} // namespace detail

template<std::size_t IterationCount, typename CallableType>
constexpr void repeat(CallableType &&Callable) {
  detail::repeat(std::make_index_sequence<IterationCount>(),
                 std::forward<CallableType>(Callable));
}

template<std::size_t IterationCount, typename CallableType>
constexpr bool repeatAnd(CallableType &&Callable) {
  return detail::repeatAnd(std::make_index_sequence<IterationCount>(),
                           std::forward<CallableType>(Callable));
}

template<std::size_t IterationCount, typename CallableType>
constexpr bool repeatOr(CallableType &&Callable) {
  return detail::repeatOr(std::make_index_sequence<IterationCount>(),
                          std::forward<CallableType>(Callable));
}

namespace examples {
using namespace std::string_view_literals;

template<std::size_t Count>
consteval std::size_t fullSize(std::array<std::string_view, Count> Components,
                               std::string_view Separator) {
  std::size_t Result = Separator.size() * Count;
  compile_time::repeat<Count>([&Result, &Components]<std::size_t Index> {
    Result += std::get<Index>(Components).size();
  });
  return Result;
}

inline constexpr std::array Components = { "instruction"sv,
                                           "0x401000:Code_x86_64"sv,
                                           "0x402000:Code_x86_64"sv,
                                           "0x403000:Code_x86_64"sv };
static_assert(fullSize(Components, "/"sv) == 75);

} // namespace examples

namespace detail {

template<std::size_t N, std::size_t I = 0>
inline constexpr bool split(std::array<std::string_view, N> &Result,
                            std::string_view Separator,
                            std::string_view Input) {
  std::size_t Position = Input.find(Separator);
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
template<std::size_t N>
inline constexpr std::optional<std::array<std::string_view, N>>
split(std::string_view Separator, std::string_view Input) {
  if (std::array<std::string_view, N> Result;
      detail::split<N>(Result, Separator, Input))
    return Result;
  else
    return std::nullopt;
}

} // namespace compile_time
