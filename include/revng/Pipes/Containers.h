#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/StringMap.h"

namespace revng::pipes {

inline constexpr char DecompileMime[] = "text/x.c+ptml+tar+gz";
inline constexpr char DecompileName[] = "decompile";
inline constexpr char DecompileExtension[] = ".c.ptml";
using DecompileStringMap = FunctionStringMap<&kinds::Decompiled,
                                             DecompileName,
                                             DecompileMime,
                                             DecompileExtension>;

} // namespace revng::pipes
