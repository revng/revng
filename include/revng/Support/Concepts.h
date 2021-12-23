#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <type_traits>

template<typename T, typename U>
concept same_as = std::is_same_v<T, U>;

template<typename T, typename U>
concept convertible_to = std::is_convertible_v<U, T>;

template<typename T>
concept TypeWithID = requires(T a) {
  { T::ID } -> convertible_to<const char &>;
};
