#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/Model/Binary.h"

namespace model {

/// Fix the model by removing invalid types.
void fixModel(TupleTree<model::Binary> &Model);

} // namespace model
