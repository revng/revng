#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/DataLayoutAnalysis/DLALayouts.h"
#include "revng/DataLayoutAnalysis/DLATypeSystem.h"
#include "revng/Model/Binary.h"
#include "revng/Model/TypeDefinition.h"

using TypeMapT = std::map<dla::LayoutTypePtr, model::UpcastableType>;

namespace dla {

/// Generate model types from a LayoutTypeSystem graph.
///\return A vector of model Types where each position corresponds to the
/// equivalence class of the LayoutTypeSystemNode that generated the type.
TypeMapT makeModelTypes(const LayoutTypeSystem &TS,
                        const LayoutTypePtrVect &Values,
                        TupleTree<model::Binary> &Model);

/// Attach model types to function arguments and return values.
/// Whether there was anything to update in the model.
bool updateFuncSignatures(const llvm::Module &M,
                          TupleTree<model::Binary> &Model,
                          const TypeMapT &TypeMap);

/// Attach model types to segments and update the model.
bool updateSegmentsTypes(const llvm::Module &M,
                         TupleTree<model::Binary> &Model,
                         const TypeMapT &TypeMap);

} // end namespace dla
