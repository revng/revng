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

// (T*)(i + k_1) -> (T*)i + k_2
// Note: k_2 = k_1 / sizeof(T)
// Constraint: k_1 % sizeof(T) == 0
def PointerArithmeticAddImmPattern
  : Pat<(Clift_BitCastOp:$res (Clift_AddOp $ptr,
                                           (Clift_ImmediateOp:$imm $k))),
        (Clift_PtrAddOp (Clift_BitCastOp $ptr, (returnType $res)),
                        (Clift_ImmediateOp (makeInteger (NativeCodeCall<[{ *pointerOffsetQuotient($0, $1) }]> $k, $res)),
                                           (returnType $imm))),
        [(Clift_AnyPointerType $res),
         (Constraint<CPred<[{ pointerOffsetQuotient($0, $1) }]>> $k, $res)]>;

// (T*)(i + n * k_1) -> (T*)i + n * k_2
// Note: k_2 = k_1 / sizeof(T)
// Constraint: k_1 % sizeof(T) == 0
def PointerArithmeticAddMulImmPattern
  : Pat<(Clift_BitCastOp:$res (Clift_AddOp $ptr,
                                           (Clift_MulOp $n,
                                                        (Clift_ImmediateOp:$imm $k)))),
        (Clift_PtrAddOp (Clift_BitCastOp $ptr, (returnType $res)),
                        (Clift_MulOp $n,
                                     (Clift_ImmediateOp (makeInteger (NativeCodeCall<[{ *pointerOffsetQuotient($0, $1) }]> $k, $res)),
                                                        (returnType $imm)))),
        [(Clift_AnyPointerType $res),
         (Constraint<CPred<[{ pointerOffsetQuotient($0, $1) }]>> $k, $res)]>;


#endif
