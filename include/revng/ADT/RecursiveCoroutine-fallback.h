#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <utility>

template<typename ReturnT = void>
using RecursiveCoroutine = ReturnT;

template<typename CoroutineT, typename... Args>
auto rc_run(CoroutineT F, Args &&... args) {
  return F(std::forward<Args>(args)...);
}

#define rc_return return

#define rc_recur
