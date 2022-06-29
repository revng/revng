#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"

namespace model {

/// Remove all the types that cannot be reached from any named type or a type
/// "outside" the type system itself.
void purgeUnnamedAndUnreachableTypes(TupleTree<model::Binary> &Model);

/// Remove all the types that cannot be reached from any type from `Functions`.
void pruneUnusedTypes(TupleTree<model::Binary> &Model);

/// Implement the purge logic.
template<bool PruneAllUnusedTypes>
void purgeTypesImpl(TupleTree<model::Binary> &Model);
} // namespace model
