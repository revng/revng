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

def BitcastBitcastPattern
  : Pat<(Clift_BitCastOp:$res (Clift_BitCastOp $arg)),
        (Clift_BitCastOp $arg)>;

def BitcastDecayPattern
  : Pat<(Clift_BitCastOp:$res (Clift_DecayOp $arg)),
        (Clift_DecayOp $arg),
        [(Clift_AnyPointerType $res)]>;

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

def ZeroExtendZeroExtendPattern
  : Pat<(Clift_ZeroExtendOp (Clift_ZeroExtendOp $arg)),
        (Clift_ZeroExtendOp $arg)>;

def SignExtendBitCastPattern
  : Pat<(Clift_SignExtendOp (Clift_BitCastOp $arg)),
        (Clift_SignExtendOp $arg),
        [(Clift_AnyIntegerType $arg)]>;

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
