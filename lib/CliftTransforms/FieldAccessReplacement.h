#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "BestTraversal.h"
#include "PointerArithmetic.h"

void replaceFieldAccess(mlir::clift::ExpressionOpInterface PointerToReplace,
                        const PointerArithmetic &Arithmetic,
                        const Traversal &BestTraversal);
