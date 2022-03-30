#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <type_traits>

template<typename T, typename U>
concept same_as = std::is_same_v<T, U>;

template<typename T, typename U>
concept convertible_to = std::is_convertible_v<T, U>;

template<typename T, typename R>
concept ConstOrNot = std::is_same_v<R, T> or std::is_same_v<const R, T>;

template<typename T>
concept Integral = std::is_integral_v<T>;

// clang-format off
namespace ranges {

template<class T>
concept range = requires(T & t) {
  std::begin(t);
  std::end(t);
};

template<class T>
concept sized_range = ranges::range<T> && requires(T & t) {
  std::size(t);
};

} // namespace ranges

// clang-format off
