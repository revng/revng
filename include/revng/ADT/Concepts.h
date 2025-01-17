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
// Concepts to simplify working with tuples.
//

template<class T>
concept TupleSizeCompatible = requires {
  std::tuple_size<T>::value;
  { std::tuple_size_v<T> } -> std::convertible_to<size_t>;
};

static_assert(TupleSizeCompatible<std::tuple<>>);
static_assert(!TupleSizeCompatible<std::vector<int>>);
static_assert(!TupleSizeCompatible<int>);

namespace revng::detail {

template<class T, size_t N>
concept TupleElementCompatibleHelper = requires(T Value) {
  typename std::tuple_element_t<N, std::remove_const_t<T>>;
  { get<N>(Value) } -> std::convertible_to<std::tuple_element_t<N, T> &>;
};

template<typename T, size_t... N>
constexpr auto checkTupleElementTypes(std::index_sequence<N...>) {
  return (TupleElementCompatibleHelper<T, N> && ...);
}

template<TupleSizeCompatible T>
constexpr auto checkAllTupleElementTypes() {
  constexpr size_t Size = std::tuple_size_v<T>;
  return checkTupleElementTypes<T>(std::make_index_sequence<Size>());
}

} // namespace revng::detail

template<class T>
concept TupleLike = (TupleSizeCompatible<T>
                     and revng::detail::checkAllTupleElementTypes<T>());

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
template<typename Type, template<typename...> class Ref>
concept SpecializationOf = requires(Type &Value) {
  []<typename... Ts>(Ref<Ts...> &) {
  }(const_cast<std::remove_const_t<Type> &>(Value));
};

template<typename T, typename Base>
concept NonBaseDerived = std::derived_from<T, Base>
                         and not std::is_same_v<T, Base>;

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
/// e.g. when instantiating traits.
template<typename Test, template<typename...> class Ref>
concept StrictSpecializationOf = revng::detail::StrictSpecialization<Test, Ref>;

//
// Other Miscellaneous concepts.
//

template<typename T, typename R>
concept ConstOrNot = std::is_same_v<R, T> or std::is_same_v<const R, T>;

template<typename T>
constexpr bool IsConstReference = std::is_const_v<std::remove_reference_t<T>>;

template<typename R, typename T>
using ConstPtrIfConst = std::conditional_t<IsConstReference<R>, const T *, T *>;

template<class R, typename ValueType>
concept RangeOf = std::ranges::range<R>
                  and std::is_convertible_v<
                    decltype(*std::declval<R>().begin()),
                    ValueType>;

template<typename T, typename... Types>
  requires(sizeof...(Types) > 0)
inline constexpr bool anyOf() {
  return (std::is_same_v<T, Types> || ...);
}
