#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>

#include "revng/Model/Binary.h"

namespace model {

/// Given \p Types, drop all the types and DynamicFunctions depending on it
///
/// Sometimes you createe a set of placeholder types in the model, but they end
/// up being invalid. In that case, they, and all the types depending on them,
/// need to be dropped.
///
/// \return the number of dropped types
unsigned dropTypesDependingOnTypes(TupleTree<model::Binary> &Binary,
                                   const std::set<const model::Type *> &Types);

/// Given an invalid model which features conflicing names, changes it to make
/// it valid
void promoteOriginalName(TupleTree<model::Binary> &Model);

/// Best effort deduplication of types that are identical
void deduplicateEquivalentTypes(TupleTree<model::Binary> &Model);

} // namespace model
