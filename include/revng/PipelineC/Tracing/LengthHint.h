#pragma once
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifdef __cplusplus
#include "revng/ADT/ConstexprString.h"

extern "C++" {
template<ConstexprString Name, int I>
inline constexpr int LengthHint = -1;
}

#define LENGTH_HINT(function_name, list_position, length_position) \
  extern "C++" {                                                   \
  template<>                                                       \
  inline constexpr int                                             \
    LengthHint<#function_name, list_position> = length_position;   \
  }
#else
#define LENGTH_HINT(function_name, list_position, length_position)
#endif
