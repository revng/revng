//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_IMMEDIATE_CANONICALIZATION_REWRITES
#define MLIR_CLIFT_IMMEDIATE_CANONICALIZATION_REWRITES

include "mlir/IR/PatternBase.td"

include "revng/Clift/Clift.td"
include "revng/CliftTransforms/ExpressionHelpers.td"

// (T*)NULL -> NULL
def BitCastNullPattern
  : Pat<(Clift_BitCastOp:$res (Clift_NullOp)),
        (Clift_NullOp),
        [(Clift_AnyPointerType $res)]>;

// (T)123 -> 123
// Constraint: T is an integer type.
def BitCastImmediatePattern
  : Pat<(Clift_BitCastOp:$res (Clift_ImmediateOp $value)),
        (Clift_ImmediateOp $value),
        [(Clift_AnyIntegerType $res)]>;

// (T)123 -> 123
def ZeroExtendImmediatePattern
  : Pat<(Clift_ZeroExtendOp:$res (Clift_ImmediateOp $value)),
        (Clift_ImmediateOp (makeInteger (NativeCodeCall<[{ $0.getValue().zext(getBitWidth($1)) }]> $value, $res)))>;

// (T)123 -> 123
def SignExtendImmediatePattern
  : Pat<(Clift_SignExtendOp:$res (Clift_ImmediateOp $value)),
        (Clift_ImmediateOp (makeInteger (NativeCodeCall<[{ $0.getValue().sext(getBitWidth($1)) }]> $value, $res)))>;

// (T)123 -> 123
def TruncateImmediatePattern
  : Pat<(Clift_TruncateOp:$res (Clift_ImmediateOp $value)),
        (Clift_ImmediateOp (makeInteger (NativeCodeCall<[{ $0.getValue().trunc(getBitWidth($1)) }]> $value, $res)))>;

#endif
