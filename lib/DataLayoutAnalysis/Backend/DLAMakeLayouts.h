#pragma once

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"

//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

namespace dla {

/// Final step, which flattens out the types into memory layouts
using LayoutPtrVector = std::vector<Layout *>;

/// Generate Layout objects from a DLATypeSystem
///
/// Some nodes of a DLATypeSystem graph can generate a Layout, that is added to
/// \a Layouts.
/// The returned vector stores pointers to the generated layouts in a specific
/// order: the index of a Layout in the vector is equal to the Node's ID
/// equivalence class.
/// Note that pointers in the returned vector may be duplicated.
///
///\param[in] TS The graph that represents layouts and their relations
///\param[out] Layouts Where to put the constructed layouts
///\return a vector of Layouts ordered using TS equivalence classes
LayoutPtrVector makeLayouts(const LayoutTypeSystem &TS, LayoutVector &Layouts);

/// Create a map between LayoutTypePtrs and Layouts
///\param Values the list of LayoutTypePtrs
///\param OrderedLayouts the list of Layouts
///\param EqClasses equivalence classes between indexes of \a Values and
///                 indexes of \a OrderedLayouts
ValueLayoutMap makeLayoutMap(const LayoutTypePtrVect &Values,
                             const LayoutPtrVector &OrderedLayouts,
                             const VectEqClasses &EqClasses);
} // namespace dla
