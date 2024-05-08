#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <cstdlib>

#include "llvm/ADT/StringRef.h"

template<typename T>
inline constexpr size_t EnumElementsCount = 0;

template<typename Enum>
inline llvm::StringRef getEnumName();
