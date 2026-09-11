//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_BOOLEAN_CAST_ELIMINATION_REWRITES
#define MLIR_CLIFT_BOOLEAN_CAST_ELIMINATION_REWRITES

include "mlir/IR/PatternBase.td"

include "revng/Clift/Clift.td"
include "revng/CliftTransforms/ExpressionHelpers.td"

def HasIntType : Constraint<CPred<[{ hasIntType($0) }]>>;

// (bool) x -> x != 0
def TestIntegerPattern
  : Pat<(Clift_TestOp:$res $arg),
        (Clift_CmpNeOp $arg,
                       (NativeCodeCall<[{ makeZeroImmediate($_builder,
                                                            $0.getDefiningOp(),
                                                            $1.getType()); }]> $res, $arg)),
        [(Clift_AnyIntegerType $arg)]>;

// (bool) x -> x != NULL
def TestPointerPattern
  : Pat<(Clift_TestOp:$res $arg),
        (Clift_CmpNeOp $arg, (Clift_NullOp (returnType (removeConst (getType $arg))))),
        [(Clift_AnyPointerType $arg)]>;

// Unlike the boolean cast, inequalities introduced by the above rewrites have
// type int. As such, any subsequent explicit extension to int can be made
// implicit.

// (int) x != y -> x != y
def IntExtendCmpNe
  : Pat<(Clift_BoolExtendOp:$res (Clift_CmpNeOp:$cmp $lhs, $rhs)),
        (Clift_ImplicitCastOp $cmp),
        [(HasIntType $res)]>;

#endif
