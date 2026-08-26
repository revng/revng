//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_EXPRESSION_POST_PROCESSING_REWRITES
#define MLIR_CLIFT_EXPRESSION_POST_PROCESSING_REWRITES

include "mlir/IR/PatternBase.td"

include "revng/Clift/Clift.td"
include "revng/CliftTransforms/ExpressionHelpers.td"

// 0xFFFFFFFF -> -1
// Constraint: the immediate value top bit and at least one other bit is set.
def NegativeImmediatePattern
  : Pat<(Clift_ImmediateOp:$res $value),
        (Clift_NegOp (Clift_ImmediateOp (makeInteger (NativeCodeCall<[{ negateIntegerValue($0.getValue()) }]> $value)),
                                        (returnType $res))),
        [(Constraint<CPred<[{ isNegativeImmediateValue($0.getValue()) }]>> $value)]>;

#endif
