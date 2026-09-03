//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_BOOLEAN_NEGATION_REWRITES
#define MLIR_CLIFT_BOOLEAN_NEGATION_REWRITES

include "mlir/IR/PatternBase.td"

include "revng/Clift/Clift.td"
include "revng/CliftTransforms/ExpressionHelpers.td"

// !(x == y) -> x != y
def NotCmpEqPattern
  : Pat<(Clift_LogicalNotOp (Clift_CmpEqOp $lhs, $rhs)),
        (Clift_CmpNeOp $lhs, $rhs)>;

// !(x != y) -> x == y
def NotCmpNePattern
  : Pat<(Clift_LogicalNotOp (Clift_CmpNeOp $lhs, $rhs)),
        (Clift_CmpEqOp $lhs, $rhs)>;

// !(x < y) -> x >= y
def NotSCmpLtPattern
  : Pat<(Clift_LogicalNotOp (Clift_SCmpLtOp $lhs, $rhs)),
        (Clift_SCmpGeOp $lhs, $rhs)>;

// !(x < y) -> x >= y
def NotUCmpLtPattern
  : Pat<(Clift_LogicalNotOp (Clift_UCmpLtOp $lhs, $rhs)),
        (Clift_UCmpGeOp $lhs, $rhs)>;

// !(x > y) -> x <= y
def NotSCmpGtPattern
  : Pat<(Clift_LogicalNotOp (Clift_SCmpGtOp $lhs, $rhs)),
        (Clift_SCmpLeOp $lhs, $rhs)>;

// !(x > y) -> x <= y
def NotUCmpGtPattern
  : Pat<(Clift_LogicalNotOp (Clift_UCmpGtOp $lhs, $rhs)),
        (Clift_UCmpLeOp $lhs, $rhs)>;

// !(x <= y) -> x > y
def NotSCmpLePattern
  : Pat<(Clift_LogicalNotOp (Clift_SCmpLeOp $lhs, $rhs)),
        (Clift_SCmpGtOp $lhs, $rhs)>;

// !(x <= y) -> x > y
def NotUCmpLePattern
  : Pat<(Clift_LogicalNotOp (Clift_UCmpLeOp $lhs, $rhs)),
        (Clift_UCmpGtOp $lhs, $rhs)>;

// !(x >= y) -> x < y
def NotSCmpGePattern
  : Pat<(Clift_LogicalNotOp (Clift_SCmpGeOp $lhs, $rhs)),
        (Clift_SCmpLtOp $lhs, $rhs)>;

// !(x >= y) -> x < y
def NotUCmpGePattern
  : Pat<(Clift_LogicalNotOp (Clift_UCmpGeOp $lhs, $rhs)),
        (Clift_UCmpLtOp $lhs, $rhs)>;

// !!x -> x
//        x ? 1 : 0
// Note: The choice of replacement depends on whether the result is
//       boolean-tested.
def LogicalNotNotPattern
  : Pat<(Clift_LogicalNotOp:$outer (Clift_LogicalNotOp $arg)),
        (forceBooleanResult $outer, $arg)>;

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

// !x ? y : z -> x ? z : y
def TernaryInversionPattern
  : Pat<(Clift_TernaryOp (Clift_LogicalNotOp $condition),
                         $lhs,
                         $rhs),
        (Clift_TernaryOp $condition, $rhs, $lhs)>;

#endif
