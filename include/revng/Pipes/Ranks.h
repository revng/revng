#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Rank.h"

namespace revng::pipes {

inline pipeline::Rank RootRank("root");

inline pipeline::Rank FunctionsRank("function", RootRank);

} // namespace revng::pipes
