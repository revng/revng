#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

template<typename ReturnT = void>
using RecursiveCoroutine = ReturnT;

#define rc_return return

#define rc_recur

#define rc_eval(...) (__VA_ARGS__)
