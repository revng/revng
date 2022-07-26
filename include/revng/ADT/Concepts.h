#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <concepts>
#include <iterator>
#include <type_traits>

template<typename T, typename U>
concept convertible_to = std::is_convertible_v<T, U>;

template<typename T>
concept NotVoid = not std::is_void_v<T>;

template<class Derived, class Base>
concept DerivesFrom = std::is_base_of_v<Base, Derived>;

template<typename T, typename R>
concept ConstOrNot = std::is_same_v<R, T> or std::is_same_v<const R, T>;

template<typename T>
concept Integral = std::is_integral_v<T>;

// clang-format off
namespace ranges {

template<class T>
concept range = requires(T &R) {
  std::begin(R);
  std::end(R);
};

template<class Container, typename ElementType>
concept typed = range<Container> &&
                std::is_convertible_v<typename Container::value_type,
                                      ElementType>;

template<class T>
concept sized_range = ranges::range<T> && requires(T &R) {
  std::size(R);
};

} // namespace ranges

// clang-format off
