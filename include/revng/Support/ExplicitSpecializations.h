#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifdef __cplusplus

#include <vector>

// Declare the existence of explicit template specializations of certain
// functions that would be otherwise heavy on build times. Make sure this file
// is included by an header that's included by all the translation units.

extern template void std::vector<unsigned int>::__push_back_slow_path<
  const unsigned int &>(const unsigned int &);

#endif
