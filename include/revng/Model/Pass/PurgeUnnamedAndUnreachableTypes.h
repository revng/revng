#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"

namespace model {

/// Remove all the types that do not verify.
void purgeInvalidTypes(TupleTree<model::Binary> &Model);

/// Remove all the types that cannot be reached from outside Binary::Types and
/// have no OriginalName or CustomName
void purgeUnnamedAndUnreachableTypes(TupleTree<model::Binary> &Model);

/// Remove all the types that cannot be reached from outside Binary::Types
void purgeUnreachableTypes(TupleTree<model::Binary> &Model);

} // namespace model
