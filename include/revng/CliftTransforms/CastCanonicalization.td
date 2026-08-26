//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_CAST_CANONICALIZATION_REWRITES
#define MLIR_CLIFT_CAST_CANONICALIZATION_REWRITES

include "mlir/IR/PatternBase.td"

include "revng/Clift/Clift.td"
include "revng/CliftTransforms/ExpressionHelpers.td"

class makeCastOp<string extend> : NativeCodeCall<[{
  makeCastOpImpl<clift::}] # extend # [{>($_builder, $1, $0)
}]>;

// test(bitcast(x)) -> test(x)
def TestBitCastPattern
  : Pat<(Clift_TestOp (Clift_BitCastOp $arg)),
        (Clift_TestOp $arg),
        [(Clift_AnyScalarType $arg)]>;

// test(bext(x)) -> x
def TestBoolExtendPattern
  : Pat<(Clift_TestOp (Clift_BoolExtendOp $arg)),
        (replaceWithValue $arg)>;

// test(zext(x)) -> test(x)
def TestZeroExtendPattern
  : Pat<(Clift_TestOp (Clift_ZeroExtendOp $arg)),
        (Clift_TestOp $arg)>;

// test(sext(x)) -> test(x)
def TestSignExtendPattern
  : Pat<(Clift_TestOp (Clift_SignExtendOp $arg)),
        (Clift_TestOp $arg)>;

def BitcastBitcastPattern
  : Pat<(Clift_BitCastOp:$res (Clift_BitCastOp $arg)),
        (Clift_BitCastOp $arg)>;

// bitcast(bext(x)) -> bext(x)
def BitcastBoolExtendPattern
  : Pat<(Clift_BitCastOp:$res (Clift_BoolExtendOp $arg)),
        (Clift_BoolExtendOp $arg),
        [(Clift_AnyIntegerType $res)]>;

def BitcastZeroExtendPattern
  : Pat<(Clift_BitCastOp:$res (Clift_ZeroExtendOp $arg)),
        (Clift_ZeroExtendOp $arg),
        [(Clift_AnyIntegerType $res)]>;

def BitcastSignExtendPattern
  : Pat<(Clift_BitCastOp:$res (Clift_SignExtendOp $arg)),
        (Clift_SignExtendOp $arg),
        [(Clift_AnyIntegerType $res)]>;

def BitcastTruncatePattern
  : Pat<(Clift_BitCastOp:$res (Clift_TruncateOp $arg)),
        (Clift_TruncateOp $arg),
        [(Clift_AnyIntegerType $res)]>;

def ZeroExtendBitCastPattern
  : Pat<(Clift_ZeroExtendOp (Clift_BitCastOp $arg)),
        (Clift_ZeroExtendOp $arg),
        [(Clift_AnyIntegerType $arg)]>;

// zext(bext(x)) -> bext(x)
def ZeroExtendBoolExtendPattern
  : Pat<(Clift_ZeroExtendOp (Clift_BoolExtendOp $arg)),
        (Clift_BoolExtendOp $arg)>;

def ZeroExtendZeroExtendPattern
  : Pat<(Clift_ZeroExtendOp (Clift_ZeroExtendOp $arg)),
        (Clift_ZeroExtendOp $arg)>;

def SignExtendBitCastPattern
  : Pat<(Clift_SignExtendOp (Clift_BitCastOp $arg)),
        (Clift_SignExtendOp $arg),
        [(Clift_AnyIntegerType $arg)]>;

// sext(bext(x)) -> bext(x)
def SignExtendBoolExtendPattern
  : Pat<(Clift_SignExtendOp (Clift_BoolExtendOp $arg)),
        (Clift_BoolExtendOp $arg)>;

def SignExtendZeroExtendPattern
  : Pat<(Clift_SignExtendOp (Clift_ZeroExtendOp $arg)),
        (Clift_ZeroExtendOp $arg)>;

def SignExtendSignExtendPattern
  : Pat<(Clift_SignExtendOp (Clift_SignExtendOp $arg)),
        (Clift_SignExtendOp $arg)>;

def TruncateBitCastPattern
  : Pat<(Clift_TruncateOp (Clift_BitCastOp $arg)),
        (Clift_TruncateOp $arg),
        [(Clift_AnyIntegerType $arg)]>;

// truncate(bext(x)) -> bext(x)
def TruncateBoolExtendPattern
  : Pat<(Clift_TruncateOp (Clift_BoolExtendOp $arg)),
        (Clift_BoolExtendOp $arg)>;

def TruncateZeroExtendPattern
  : Pat<(Clift_TruncateOp:$res (Clift_ZeroExtendOp $arg)),
        (makeCastOp<"ZeroExtendOp"> $res, $arg)>;

def TruncateSignExtendPattern
  : Pat<(Clift_TruncateOp:$res (Clift_SignExtendOp $arg)),
        (makeCastOp<"SignExtendOp"> $res, $arg)>;

def TruncateTruncatePattern
  : Pat<(Clift_TruncateOp (Clift_TruncateOp $arg)),
        (Clift_TruncateOp $arg)>;

#endif
