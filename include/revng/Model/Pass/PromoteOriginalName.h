#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"

namespace model {

/// Given an invalid model which features conflicing names, changes it to make
/// it valid
void promoteOriginalName(TupleTree<model::Binary> &Model);

} // namespace model
