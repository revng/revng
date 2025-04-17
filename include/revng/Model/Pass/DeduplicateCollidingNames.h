#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"

namespace model {

/// This looks for groups of names are duplicates of each other and appends
/// arbitrary suffixes to prevent collisions.
void deduplicateCollidingNames(TupleTree<model::Binary> &Binary);

} // namespace model
