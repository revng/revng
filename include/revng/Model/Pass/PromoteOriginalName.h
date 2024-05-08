#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/Model/Binary.h"

namespace model {

/// Promote OriginalName fields to CustomName ensuring
/// the validity of the model is preserved
void promoteOriginalName(TupleTree<model::Binary> &Model);

} // namespace model
