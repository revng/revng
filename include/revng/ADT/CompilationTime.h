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

/// Calls \ref Callable with unpacked sequence of \ref IterationCount.
/// Example:
/// ```cpp
/// compile_time::callWithIndexSequence<3>([]<size_t ...I>() {
///   // The parameter pack I is composed of 0, 1, 2 in this example
/// });
/// ```
template<size_t IterationCount, typename CallableType>
constexpr auto callWithIndexSequence(CallableType &&Callable) {
  auto Runner = [&Callable]<size_t... I>(std::index_sequence<I...>) {
    return Callable.template operator()<I...>();
  };
  return Runner(std::make_index_sequence<IterationCount>{});
}

/// Calls \ref Callable with unpacked sequence of the size of tuple-like
/// \ref TupleType. See the documentation for the size_t counterpart for usage.
template<TupleSizeCompatible TupleType, typename CallableType>
constexpr auto callWithIndexSequence(CallableType &&Callable) {
  return callWithIndexSequence<std::tuple_size_v<TupleType>>(Callable);
}

namespace detail {

template<typename T>
using RVHelper = decltype(std::declval<T>().template operator()<0>());

template<size_t IterationCount, typename CallableType>
  requires(IterationCount > 0)
constexpr auto repeat(CallableType &&Callable) {
  return callWithIndexSequence<IterationCount>([&Callable]<size_t... I>() {
    if constexpr (std::is_same_v<RVHelper<CallableType>, void>)
      (Callable.template operator()<I>(), ...);
    else
      return std::tie(Callable.template operator()<I>()...);
  });
}

} // namespace detail

/// Calls \ref Callable \ref IterationCount times.
template<size_t IterationCount, typename CallableType>
constexpr auto repeat(CallableType &&Callable) {
  if constexpr (IterationCount == 0)
    return;
  else
    return detail::repeat<IterationCount>(std::forward<CallableType>(Callable));
}

/// Calls \ref Callable \ref IterationCount times, while applying logical AND
/// operation to the return values.
template<size_t IterationCount, typename CallableType>
constexpr bool repeatAnd(CallableType &&Callable) {
  return callWithIndexSequence<IterationCount>([&Callable]<size_t... I>() {
    return (Callable.template operator()<I>() && ...);
  });
}

/// Calls \ref Callable \ref IterationCount times, while applying logical OR
/// operation to the return values.
template<size_t IterationCount, typename CallableType>
constexpr bool repeatOr(CallableType &&Callable) {
  return callWithIndexSequence<IterationCount>([&Callable]<size_t... I>() {
    return (Callable.template operator()<I>() || ...);
  });
}

/// Calls \ref Callable \ref IterationCount times, returns the amount of times
/// the \ref Callable returned a truthy value.
template<size_t IterationCount, typename CallableType>
constexpr size_t count(CallableType &&Callable) {
  return callWithIndexSequence<IterationCount>([&Callable]<size_t... I>() {
    return ((Callable.template operator()<I>() ? 1 : 0) + ...);
  });
}

/// Calls \ref Callable \ref IterationCount times, makes sure at most one of
/// those invocations has a non-zero return value, then returns its index if
/// there is one, or `std::nullopt` if there's none.
template<size_t IterationCount, typename CallableType>
constexpr std::optional<size_t> select(CallableType &&Callable) {
  return callWithIndexSequence<IterationCount>([&Callable]<size_t... I>() {
    std::array<bool, IterationCount> Results;
    ((Results[I] = Callable.template operator()<I>()), ...);
    if constexpr (std::ranges::count(Results, true) == 1) {
      auto It = std::ranges::find(Results, true);
      return std::distance(Results.begin(), It);
    } else {
      return std::nullopt;
    }
  });
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
