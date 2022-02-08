#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Kind.h"

namespace revng::pipes {
extern pipeline::Rank RootRank;

extern pipeline::Kind CFepper;
extern pipeline::Kind Binary;

extern pipeline::Kind Object;
extern pipeline::Kind Translated;

extern pipeline::Kind ABIEnforced;

extern pipeline::Kind Dead;

} // namespace revng::pipes
