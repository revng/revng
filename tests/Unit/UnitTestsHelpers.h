#ifndef UNITTESTSHELPERS_H
#define UNITTESTSHELPERS_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <exception>

// Local libraries includes
#include "revng/Support/Assert.h"

inline void boost::throw_exception(std::exception const &E) {
  revng_abort(E.what());
}

#endif // UNITTESTSHELPERS_H
