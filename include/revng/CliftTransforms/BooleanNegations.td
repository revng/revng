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
def NotCmpLtPattern
  : Pat<(Clift_LogicalNotOp (Clift_CmpLtOp $lhs, $rhs)),
        (Clift_CmpGeOp $lhs, $rhs)>;

// !(x > y) -> x <= y
def NotCmpGtPattern
  : Pat<(Clift_LogicalNotOp (Clift_CmpGtOp $lhs, $rhs)),
        (Clift_CmpLeOp $lhs, $rhs)>;

// !(x <= y) -> x > y
def NotCmpLePattern
  : Pat<(Clift_LogicalNotOp (Clift_CmpLeOp $lhs, $rhs)),
        (Clift_CmpGtOp $lhs, $rhs)>;

// !(x >= y) -> x < y
def NotCmpGePattern
  : Pat<(Clift_LogicalNotOp (Clift_CmpGeOp $lhs, $rhs)),
        (Clift_CmpLtOp $lhs, $rhs)>;

// !!x -> x
//        x ? 1 : 0
// Note: The choice of replacement depends on whether the result is
//       boolean-tested.
def LogicalNotNotPattern
  : Pat<(Clift_LogicalNotOp:$outer (Clift_LogicalNotOp $arg)),
        (forceBooleanResult $outer, $arg)>;

#endif
