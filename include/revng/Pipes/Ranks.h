#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Rank.h"

namespace revng::pipes::ranks {

inline pipeline::Rank Root("root");

inline pipeline::Rank Function("function", Root);

} // namespace revng::pipes::ranks
