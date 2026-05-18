//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_EXPRESSION_REWRITES
#define MLIR_CLIFT_EXPRESSION_REWRITES

include "mlir/IR/PatternBase.td"

include "revng/Clift/Clift.td"
include "revng/CliftTransforms/ExpressionHelpers.td"

//===------------------------- Boolean expressions -------------------------===//

// x == 0 -> !x
def CmpEqWithZeroPattern
  : Pat<(Clift_CmpEqOp:$res $arg, (Clift_ImmediateOp $value)),
        (Clift_LogicalNotOp $arg),
        [(Constraint<CPred<"$0.getValue().isZero()">> $value)]>;

// x != 0 -> x
//           x ? 1 : 0
// Note: The choice of replacement depends on whether the result is
//       boolean-tested.
def CmpNeWithZeroPattern
  : Pat<(Clift_CmpNeOp:$res $arg, (Clift_ImmediateOp $value)),
        (forceBooleanResult $res, $arg),
        [(Constraint<CPred<"$0.getValue().isZero()">> $value)]>;

// !x && !y -> x || y
def DeMorganConjunctionPattern
  : Pat<(Clift_LogicalNotOp (Clift_LogicalAndOp (Clift_LogicalNotOp $lhs),
                                                (Clift_LogicalNotOp $rhs))),
        (Clift_LogicalOrOp $lhs, $rhs)>;

// !x || !y -> x && y
def DeMorganDisjunctionPattern
  : Pat<(Clift_LogicalNotOp (Clift_LogicalOrOp (Clift_LogicalNotOp $lhs),
                                               (Clift_LogicalNotOp $rhs))),
        (Clift_LogicalAndOp $lhs, $rhs)>;

//===--------------------------- Math identities --------------------------===//

// x ^ 0xffffffff -> ~x
def XorAllOnesPattern
  : Pat<(Clift_BitwiseXorOp $arg, (Clift_ImmediateOp:$imm $value)),
        (Clift_BitwiseNotOp $arg),
        [(Constraint<CPred<"areAllBitsSet($0.getValue(), $1.getType())">> $value, $imm)]>;

//===---------------------------- Miscellaneous ---------------------------===//

// (T)123 -> 123
// Constraint: T is a primitive integer type
def BitCastImmediatePattern
  : Pat<(Clift_BitCastOp:$res (Clift_ImmediateOp $value)),
        (Clift_ImmediateOp (makeInteger (NativeCodeCall<"truncateIntegerValue($0, $1)"> $value, $res))),
        [(Clift_AnyPrimitiveIntegerType $res)]>;

// (T)123 -> 123
// Constraint: T is a primitive integer type
def ExtendImmediatePattern
  : Pat<(Clift_ExtendOp:$res (Clift_ImmediateOp $value)),
        (Clift_ImmediateOp (makeInteger (NativeCodeCall<"truncateIntegerValue($0, $1)"> $value, $res))),
        [(Clift_AnyPrimitiveIntegerType $res)]>;

// (T)123 -> 123
// Constraint: T is a primitive integer type
def TruncateImmediatePattern
  : Pat<(Clift_TruncateOp:$res (Clift_ImmediateOp $value)),
        (Clift_ImmediateOp (makeInteger (NativeCodeCall<"truncateIntegerValue($0, $1)"> $value, $res))),
        [(Clift_AnyPrimitiveIntegerType $res)]>;

// (enum E)123 -> A
// Constraint: A is an enumerator of E whose value is 123.
def EnumImmediatePattern
  : Pat<(Clift_BitCastOp:$res (Clift_ImmediateOp $value)),
        (Clift_ImmediateOp $value),
        [(Constraint<CPred<[{ hasEnumeratorValue($0.getType(),
                                                 $1.getValue().getZExtValue()) }]>> $res, $value)]>;

// *&x -> x
def IndirectionAddressofPattern
  : Pat<(Clift_IndirectionOp (Clift_AddressofOp $arg)),
        (replaceWithValue $arg)>;

// &*x -> x
def AddressofIndirectionPattern
  : Pat<(Clift_AddressofOp (Clift_IndirectionOp $arg)),
        (replaceWithValue $arg)>;

// (T)x -> x
// Constraint: the type of x is equivalent to T.
def RedundantBitCastPattern
  : Pat<(Clift_BitCastOp:$res $arg),
        (replaceWithValue $arg),
        [(TypesAreEquivalent $res, $arg)]>;

// *(T*)p = x ->         *p = (T)x
//            -> *(T*)&( *p = (T)x )
// Note: The choice of replacement depends on whether the result is discarded.
// Constraint: p is of type U* and x is convertible to type U using a bitcast.
def AssignTypePunnedPattern
  : Pattern<(Clift_AssignOp:$old_res (Clift_IndirectionOp:$old_ref (Clift_BitCastOp:$new_ptr $ptr)),
                                     $value),
            [(Clift_AssignOp:$new_res (Clift_IndirectionOp $ptr),
                                      (Clift_BitCastOp $value,
                                                       (returnType (getPointeeType $ptr)))),
             (NativeCodeCall<"assignTypePunnedResult($_builder, $0, $1, $2, $3)"> $old_res,
                                                                                  $new_res,
                                                                                  $new_ptr,
                                                                                  $old_ref)],
            [(Constraint<CPred<"assignTypePunnedConstraint($0, $1)">> $ptr, $value)]>;

// (T*)(i + k_1) -> (T*)i + k_2
// Note: k_2 = k_1 / sizeof(T)
// Constraint: k_1 % sizeof(T) == 0
def PointerArithmeticAddImmPattern
  : Pat<(Clift_BitCastOp:$res (Clift_AddOp $ptr,
                                           (Clift_ImmediateOp:$imm $k))),
        (Clift_PtrAddOp (Clift_BitCastOp $ptr, (returnType $res)),
                        (Clift_ImmediateOp (makeInteger (NativeCodeCall<"ptrOffsetDivMod($0, $1).Div"> $k, $res)),
                                           (returnType $imm))),
        [(Clift_AnyPointerType $res),
         (Constraint<CPred<"ptrOffsetDivMod($0, $1).Mod == 0">> $k, $res)]>;

// (T*)(i + n * k_1) -> (T*)i + n * k_2
// Note: k_2 = k_1 / sizeof(T)
// Constraint: k_1 % sizeof(T) == 0
def PointerArithmeticAddMulImmPattern
  : Pat<(Clift_BitCastOp:$res (Clift_AddOp $ptr,
                                           (Clift_MulOp $n,
                                                        (Clift_ImmediateOp:$imm $k)))),
        (Clift_PtrAddOp (Clift_BitCastOp $ptr, (returnType $res)),
                        (Clift_MulOp $n,
                                     (Clift_ImmediateOp (makeInteger (NativeCodeCall<"ptrOffsetDivMod($0, $1).Div"> $k, $res)),
                                                        (returnType $imm)))),
        [(Clift_AnyPointerType $res),
         (Constraint<CPred<"ptrOffsetDivMod($0, $1).Mod == 0">> $k, $res)]>;

// (*x).m -> x->m
def IndirectAccessPattern
  : Pat<(Clift_AccessOp (Clift_IndirectionOp $arg), $indirect, $member_index),
        (Clift_AccessOp $arg, (NativeCodeCall<"$_builder.getUnitAttr()">), $member_index),
        [(Constraint<CPred<"not $0">> $indirect)]>;

// (&x)->m -> x.m
def DirectAccessPattern
  : Pat<(Clift_AccessOp (Clift_AddressofOp $arg), $indirect, $member_index),
        (Clift_AccessOp $arg, (NativeCodeCall<"mlir::UnitAttr(nullptr)">), $member_index),
        [(Constraint<CPred<"$0">> $indirect)]>;

// x[0] -> *x
// Constraint: the subscript index is a constant zero.
def SubscriptZeroPattern
  : Pat<(Clift_SubscriptOp $ptr, (Clift_ImmediateOp $value)),
        (Clift_IndirectionOp $ptr),
        [(Constraint<CPred<"$0.getValue().isZero()">> $value)]>;


#endif
