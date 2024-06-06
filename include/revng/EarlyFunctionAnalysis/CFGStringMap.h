#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/StringMap.h"

namespace revng::pipes {

inline constexpr char CFGMime[] = "text/yaml+tar+gz";
inline constexpr char CFGName[] = "cfg";
inline constexpr char CFGExtension[] = ".yml";

using CFGMap = FunctionStringMap<&kinds::CFG, CFGName, CFGMime, CFGExtension>;
} // namespace revng::pipes
