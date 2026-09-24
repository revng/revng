//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_EXPRESSION_OPTIONAL_REWRITES
#define MLIR_CLIFT_EXPRESSION_OPTIONAL_REWRITES

include "mlir/IR/PatternBase.td"

include "revng/Clift/Clift.td"
include "revng/CliftTransforms/ExpressionHelpers.td"

// p[0] -> *p
// Constraint: the subscript index is zero.
def SubscriptZeroPattern
  : Pat<(Clift_SubscriptOp $ptr, (Clift_ImmediateOp $index)),
        (Clift_IndirectionOp $ptr),
        [(Constraint<CPred<[{ $0.getValue().isZero() }]>> $index)]>;

#endif
