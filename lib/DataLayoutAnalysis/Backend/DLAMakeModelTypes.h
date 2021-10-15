#pragma once

//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Model/Type.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"

using WritableModelT = TupleTree<model::Binary>;
using TypeVect = std::vector<model::QualifiedType>;
using TypeMapT = std::map<dla::LayoutTypePtr, model::QualifiedType>;

namespace dla {

///\brief Generate model types from a LayoutTypeSystem graph.
///\return A vector of model Types where each position corresponds to the
/// equivalence class of the LayoutTypeSystemNode that generated the type.
TypeMapT makeModelTypes(const LayoutTypeSystem &TS,
                        const LayoutTypePtrVect &Values,
                        WritableModelT &Model);

///\brief Attach model types to function arguments and return values.
///\brief Whether there was anything to update in the model.
bool updateFuncSignatures(const llvm::Module &M,
                          WritableModelT &Model,
                          const TypeMapT &TypeMap);

} // end namespace dla