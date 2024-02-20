#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>

#include "llvm/Support/ManagedStatic.h"

#include "revng/Model/Binary.h"

namespace model {

/// Given \p Defs, drop all the types and dynamic functions that depend on them
///
/// Sometimes you create a set of placeholder types in the model, but they end
/// up being invalid. In that case, they, and all the types depending on them,
/// need to be dropped.
///
/// \return the number of dropped types
unsigned
dropTypesDependingOnDefinitions(TupleTree<model::Binary> &Binary,
                                const std::set<const TypeDefinition *> &Defs);

} // namespace model
