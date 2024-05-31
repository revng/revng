#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <tuple>

/// Tuple helpers to be used in type definitions
template<typename T1>
using make_tuple_t = decltype(std::make_tuple(std::declval<T1>()));

template<typename T1, typename T2>
using tuple_cat_t = decltype(std::tuple_cat(std::declval<T1>(),
                                            std::declval<T2>()));
