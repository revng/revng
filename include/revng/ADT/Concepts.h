#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <concepts>
#include <iterator>
#include <string>
#include <string_view>
#include <type_traits>

//
// The concepts from the STL.
// TODO: remove these after updating the libc++ version.
//

// clang-format off
template<class Derived, class Base>
concept derived_from = std::is_base_of_v<Base, Derived>
                       && std::is_convertible_v<const volatile Derived *,
                                                const volatile Base *>;
// clang-format on

template<typename T, typename U>
concept convertible_to = std::is_convertible_v<T, U>;

template<typename T>
concept integral = std::is_integral_v<T>;

template<class T>
concept equality_comparable = requires(T &&LHS, T &&RHS) {
  { LHS == RHS } -> convertible_to<bool>;
};

/// TODO: Remove after updating to clang-format with concept support.
struct ClangFormatPleaseDoNotBreakMyCode;
// clang-format off
// clang-format on

namespace ranges {

template<class T>
concept range = requires(T &&R) {
  std::begin(R);
  std::end(R);
};

template<class T>
concept sized_range = ranges::range<T> && requires(T &&R) {
  std::size(R);
};

} // namespace ranges

//
// Concepts to simplify working with tuples.
//

template<class T>
concept TupleSizeCompatible = requires {
  std::tuple_size<T>::value;
  { std::tuple_size_v<T> } -> convertible_to<size_t>;
};

static_assert(TupleSizeCompatible<std::tuple<>>);
static_assert(!TupleSizeCompatible<std::vector<int>>);
static_assert(!TupleSizeCompatible<int>);

namespace revng::detail {

template<class T, std::size_t N>
concept TupleElementCompatibleHelper = requires(T Value) {
  typename std::tuple_element_t<N, std::remove_const_t<T>>;
  { get<N>(Value) } -> convertible_to<std::tuple_element_t<N, T> &>;
};

template<typename T, size_t... N>
constexpr auto checkTupleElementTypes(std::index_sequence<N...>) {
  return (TupleElementCompatibleHelper<T, N> && ...);
}

template<TupleSizeCompatible T>
constexpr auto checkAllTupleElementTypes() {
  constexpr std::size_t Size = std::tuple_size_v<T>;
  return checkTupleElementTypes<T>(std::make_index_sequence<Size>());
}

} // namespace revng::detail

// clang-format off
template<class T>
concept TupleLike = (TupleSizeCompatible<T>
                     and revng::detail::checkAllTupleElementTypes<T>());
// clang-format on

static_assert(TupleLike<std::tuple<>>);
static_assert(TupleLike<std::tuple<int, int, long>>);
static_assert(TupleLike<std::pair<int, int>>);
static_assert(TupleLike<std::array<int, 0>>);
static_assert(not TupleLike<std::vector<int>>);

//
// Concepts to simplify working with specializations of templates.
//

/// A concept that helps determine whether a given object is a specialization
/// (or is inheriting a specialization) of a given template.
///
/// This lets us cut down on the number of `IsX` (e.g. `IsGenericGraph`,
/// `IsRank`) concepts needed since we can now just say
/// `SpecializationOf<GenericGraph>` on the interface boundaries and get
/// the expected check.
///
/// TODO: this requires clang-13+, uncomment it after the update.
// template<typename Type, template<typename...> class Ref>
// concept SpecializationOf = requires(Type &&Value) {
//   []<typename... Ts>(Ref<Ts...> &){}(Value);
// };
//
// static_assert(SpecializationOf<std::pair<int, long>, std::pair>);
// static_assert(SpecializationOf<const std::pair<int, long>, std::pair>);
// static_assert(SpecializationOf<std::string, std::basic_string>);
// static_assert(SpecializationOf<std::ostream, std::basic_ios>);
// static_assert(not SpecializationOf<std::string, std::basic_string_view>);

/// TODO: Remove after updating to clang-format with concept support.
struct ClangFormatPleaseDoNotBreakMyCode;
// clang-format off
// clang-format on

namespace revng::detail {

template<typename Test, template<typename...> class Ref>
struct StrictSpecializationHelper : std::false_type {};

template<template<typename...> class Ref, typename... Args>
struct StrictSpecializationHelper<Ref<Args...>, Ref> : std::true_type {};

template<template<typename...> class Ref, typename... Args>
struct StrictSpecializationHelper<const Ref<Args...>, Ref> : std::true_type {};

template<typename Test, template<typename...> class Ref>
constexpr bool
  StrictSpecialization = StrictSpecializationHelper<Test, Ref>::value;

} // namespace revng::detail

/// A more strict version of the `SpecializationOf` concept. This one only
/// allows direct specializations (and aliases) - no inheritance.
///
/// It's useful in the cases when template parameter deduction is important,
/// e.g. when instanciating traits.
template<typename Test, template<typename...> class Ref>
concept StrictSpecializationOf = revng::detail::StrictSpecialization<Test, Ref>;

static_assert(StrictSpecializationOf<std::pair<int, long>, std::pair>);
static_assert(StrictSpecializationOf<const std::pair<int, long>, std::pair>);
static_assert(StrictSpecializationOf<std::string, std::basic_string>);
static_assert(not StrictSpecializationOf<std::ostream, std::basic_ios>);
static_assert(not StrictSpecializationOf<std::string, std::basic_string_view>);

//
// Other Miscellanious concepts.
//

template<typename T, typename R>
concept ConstOrNot = std::is_same_v<R, T> or std::is_same_v<const R, T>;

// clang-format off
namespace ranges {
template<class Range, typename ValueType>
concept range_with_value_type = ranges::range<Range> &&
                std::is_convertible_v<typename Range::value_type,
                                      ValueType>;
} // namespace ranges
// clang-format on
