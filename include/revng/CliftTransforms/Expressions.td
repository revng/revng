//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_EXPRESSION_REWRITES
#define MLIR_CLIFT_EXPRESSION_REWRITES

include "mlir/IR/PatternBase.td"

include "revng/Clift/Clift.td"
include "revng/CliftTransforms/ExpressionHelpers.td"

//===------------------------- Boolean expressions -------------------------===//

// x == 0 -> !test(x)
def CmpEqWithZeroPattern
  : Pat<(Clift_CmpEqOp $arg, (Clift_ImmediateOp $value)),
        (Clift_LogicalNotOp (Clift_TestOp $arg)),
        [(Constraint<CPred<"$0.getValue().isZero()">> $value)]>;

// x == NULL -> !test(x)
def CmpEqWithNullPattern
  : Pat<(Clift_CmpEqOp $arg, (Clift_NullOp)),
        (Clift_LogicalNotOp (Clift_TestOp $arg))>;

// x != 0 -> test(x)
def CmpNeWithZeroPattern
  : Pat<(Clift_CmpNeOp $arg, (Clift_ImmediateOp $value)),
        (Clift_TestOp $arg),
        [(Constraint<CPred<"$0.getValue().isZero()">> $value)]>;

// x != NULL -> test(x)
def CmpNeWithNullPattern
  : Pat<(Clift_CmpNeOp $arg, (Clift_NullOp)),
        (Clift_TestOp $arg)>;

//===--------------------------- Math identities --------------------------===//

// x ^ 0xffffffff -> ~x
def XorAllOnesPattern
  : Pat<(Clift_BitwiseXorOp $arg, (Clift_ImmediateOp:$imm $value)),
        (Clift_BitwiseNotOp $arg),
        [(Constraint<CPred<[{ $0.getValue().isAllOnes() }]>> $value, $imm)]>;

// 0 - x -> -x
def ZeroSubPattern
  : Pat<(Clift_SubOp (Clift_ImmediateOp $value), $arg),
        (Clift_NegOp $arg),
        [(Constraint<CPred<[{ $0.getValue().isZero() }]>> $value)]>;

//===---------------------------- Miscellaneous ---------------------------===//

// (T *)0 -> NULL
def NullImmediatePattern
  : Pat<(Clift_BitCastOp:$res (Clift_ImmediateOp $value)),
        (Clift_NullOp),
        [(Clift_AnyPointerType $res),
         (Constraint<CPred<[{ $0.getValue().isZero() }]>> $value)]>;

// *&x -> x
// Constraint: the type of x is equivalent to the type of the original expression.
def IndirectionAddressofPattern
  : Pat<(Clift_IndirectionOp:$res (Clift_AddressofOp $arg)),
        (replaceWithValue $arg),
        [(TypesAreEquivalent $res, $arg)]>;

// &*x -> x
// Constraint: the type of x is equivalent to the type of the original expression.
def AddressofIndirectionPattern
  : Pat<(Clift_AddressofOp:$res (Clift_IndirectionOp $arg)),
        (replaceWithValue $arg),
        [(TypesAreEquivalent $res, $arg)]>;

// (T)x -> x
// Constraint: the type of x is equivalent to T.
def RedundantBitCastPattern
  : Pat<(Clift_BitCastOp:$res $arg),
        (replaceWithValue $arg),
        [(TypesAreEquivalent $res, $arg)]>;

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

// (*x).m -> x->m
def IndirectAccessPattern
  : Pat<(Clift_DirectAccessOp (Clift_IndirectionOp $arg), $member_index),
        (Clift_IndirectAccessOp $arg, $member_index)>;

// (&x)->m -> x.m
def DirectAccessPattern
  : Pat<(Clift_IndirectAccessOp (Clift_AddressofOp $arg), $member_index),
        (Clift_DirectAccessOp $arg, $member_index)>;

#endif
