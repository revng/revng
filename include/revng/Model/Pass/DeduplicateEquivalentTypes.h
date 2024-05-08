#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/Model/Binary.h"

namespace model {

/// Best effort deduplication of types that are structurally equivalent.
void deduplicateEquivalentTypes(TupleTree<model::Binary> &Model);

} // namespace model
