#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <exception>

#include "revng/Support/Assert.h"

inline void boost::throw_exception(std::exception const &E) {
  revng_abort(E.what());
}
