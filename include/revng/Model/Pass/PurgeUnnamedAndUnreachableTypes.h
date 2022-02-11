#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"

namespace model {

/// Remove all the types that cannot be reached from
/// any named type or a type "outside" the type system itself.
void purgeUnnamedAndUnreachableTypes(TupleTree<model::Binary> &Model);

} // namespace model
