//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_EXPRESSION_REWRITES
#define MLIR_CLIFT_EXPRESSION_REWRITES

include "mlir/IR/PatternBase.td"

include "revng/Clift/Clift.td"
include "revng/CliftTransforms/ExpressionHelpers.td"

//===------------------------- Boolean expressions -------------------------===//

// test(bext(x)) -> x
def TestBoolExtendPattern
  : Pat<(Clift_TestOp (Clift_BoolExtendOp $arg)),
        (replaceWithValue $arg)>;

// test(bitcast(x)) -> test(x)
def TestBitCastPattern
  : Pat<(Clift_TestOp (Clift_BitCastOp $arg)),
        (Clift_TestOp $arg),
        [(Clift_AnyScalarType $arg)]>;

// test(zext(x)) -> test(x)
def TestZeroExtendPattern
  : Pat<(Clift_TestOp (Clift_ZeroExtendOp $arg)),
        (Clift_TestOp $arg)>;

// test(sext(x)) -> test(x)
def TestSignExtendPattern
  : Pat<(Clift_TestOp (Clift_SignExtendOp $arg)),
        (Clift_TestOp $arg)>;

// bitcast(bext(x)) -> bext(x)
def BitcastBoolExtendPattern
  : Pat<(Clift_BitCastOp:$res (Clift_BoolExtendOp $arg)),
        (Clift_BoolExtendOp $arg),
        [(Clift_AnyIntegerType $res)]>;

// zext(bext(x)) -> bext(x)
def ZeroExtendBoolExtendPattern
  : Pat<(Clift_ZeroExtendOp (Clift_BoolExtendOp $arg)),
        (Clift_BoolExtendOp $arg)>;

// sext(bext(x)) -> bext(x)
def SignExtendBoolExtendPattern
  : Pat<(Clift_SignExtendOp (Clift_BoolExtendOp $arg)),
        (Clift_BoolExtendOp $arg)>;

// truncate(bext(x)) -> bext(x)
def TruncateBoolExtendPattern
  : Pat<(Clift_TruncateOp (Clift_BoolExtendOp $arg)),
        (Clift_BoolExtendOp $arg)>;

// x == 0 -> !test(x)
def CmpEqWithZeroPattern
  : Pat<(Clift_CmpEqOp $arg, (Clift_ImmediateOp $value)),
        (Clift_LogicalNotOp (Clift_TestOp $arg)),
        [(Constraint<CPred<"$0.getValue().isZero()">> $value)]>;

// x != 0 -> test(x)
def CmpNeWithZeroPattern
  : Pat<(Clift_CmpNeOp $arg, (Clift_ImmediateOp $value)),
        (Clift_TestOp $arg),
        [(Constraint<CPred<"$0.getValue().isZero()">> $value)]>;

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
