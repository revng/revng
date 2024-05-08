#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <set>

#include "llvm/Support/ManagedStatic.h"

#include "revng/Model/Binary.h"

namespace model {

/// Given \p Types, drop all the types and DynamicFunctions depending on it
///
/// Sometimes you create a set of placeholder types in the model, but they end
/// up being invalid. In that case, they, and all the types depending on them,
/// need to be dropped.
///
/// \return the number of dropped types
unsigned dropTypesDependingOnTypes(TupleTree<model::Binary> &Binary,
                                   const std::set<const model::Type *> &Types);

} // namespace model
