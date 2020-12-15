#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#if defined(DISABLE_RECURSIVE_COROUTINES)

#include "RecursiveCoroutine-fallback.h"

#else

#include "RecursiveCoroutine-coroutine.h"

#endif
