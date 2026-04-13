#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "BestTraversal.h"
#include "PointerArithmetic.h"

/// Perform the replacement of the pointer arithmetic access
/// (`PointerToReplace`), with operations equivalent to the `BestTraversal`.
/// Returns `true` if a type was propagated through an `indirection`, signaling
/// that a subsequent EFA pass may find new rewriting opportunities.
bool replaceFieldAccess(clift::ExpressionOpInterface PointerToReplace,
                        const PointerArithmetic &Arithmetic,
                        const Traversal &BestTraversal);
