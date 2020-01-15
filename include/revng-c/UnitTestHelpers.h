#ifndef REVNG_UNITTESTHELPERS_H
#define REVNG_UNITTESTHELPERS_H

// NOTE: this file was copy-pasted directly from revng. It is intended to
// silence errors with boost test when compiling with -fno-exception.
// We should switch to using the UnitTestHelpers.h version shipped by revng as
// soon as revng starts installing it.

//
// This file is distributed under the MIT Licens. See License.md for details.
//

// Standard includes
#include <exception>

// revng includes
#include <revng/Support/Assert.h>

inline void boost::throw_exception(std::exception const &E) {
  revng_abort(E.what());
}

#endif // REVNG_UNITTESTHELPERS_H
