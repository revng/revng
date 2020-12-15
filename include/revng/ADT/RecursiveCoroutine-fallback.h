#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

template<typename ReturnT = void>
using RecursiveCoroutine = ReturnT;

template<typename CoroutineT, typename... Args>
auto rc_run(CoroutineT F, Args... Arguments) {
  return F(Arguments...);
}

#define rc_return return

#define rc_recur
