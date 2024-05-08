#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/Model/Binary.h"

namespace model {

inline constexpr const char *AddPrimitiveTypesFlag = "add-primitive-types";

/// Adds all the required model::PrimitiveTypes to the Model, if necessary
void addPrimitiveTypes(TupleTree<model::Binary> &Model);

} // namespace model
