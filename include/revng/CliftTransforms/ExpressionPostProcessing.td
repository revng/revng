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

//===-------------------------- Cast elimination --------------------------===//

class UnaryCastElimination<Op operation>
  : Pat<(operation (Clift_BitCastOp $arg)),
        (Clift_BitCastOp $arg),
        [(Clift_AnyIntegerType $arg)]>;

class BinaryCastElimination<Op operation>
  : Pat<(operation (Clift_BitCastOp $lhs), (Clift_BitCastOp $rhs)),
        (Clift_BitCastOp (operation $lhs, $rhs)),
        [(Clift_AnyIntegerType $lhs), (TypesAreEquivalent $lhs, $rhs)]>;

class ShiftCastElimination<Op operation>
  : Pat<(operation (Clift_BitCastOp $lhs), $rhs),
        (Clift_BitCastOp (operation $lhs, $rhs)),
        [(Clift_AnyIntegerType $lhs)]>;

def NegCastElimination : UnaryCastElimination<Clift_NegOp>;
def AddCastElimination : BinaryCastElimination<Clift_AddOp>;
def SubCastElimination : BinaryCastElimination<Clift_SubOp>;
def MulCastElimination : BinaryCastElimination<Clift_MulOp>;
def SDivCastElimination : BinaryCastElimination<Clift_SDivOp>;
def UDivCastElimination : BinaryCastElimination<Clift_UDivOp>;
def SRemCastElimination : BinaryCastElimination<Clift_SRemOp>;
def URemCastElimination : BinaryCastElimination<Clift_URemOp>;

def ShlCastElimination : ShiftCastElimination<Clift_ShlOp>;
def ShrCastElimination : ShiftCastElimination<Clift_ShrOp>;
def SarCastElimination : ShiftCastElimination<Clift_SarOp>;

def BitwiseNotCastElimination : UnaryCastElimination<Clift_BitwiseNotOp>;
def BitwiseAndCastElimination : BinaryCastElimination<Clift_BitwiseAndOp>;
def BitwiseOrCastElimination : BinaryCastElimination<Clift_BitwiseOrOp>;
def BitwiseXorCastElimination : BinaryCastElimination<Clift_BitwiseXorOp>;

def TernaryCastElimination
  : Pat<(Clift_TernaryOp $condition,
                         (Clift_BitCastOp $lhs),
                         (Clift_BitCastOp $rhs)),
        (Clift_BitCastOp (Clift_TernaryOp $condition, $lhs, $rhs)),
        [(TypesAreEquivalent $lhs, $rhs)]>;

#endif
