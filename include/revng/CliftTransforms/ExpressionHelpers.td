//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_EXPRESSION_HELPERS
#define MLIR_CLIFT_EXPRESSION_HELPERS

include "mlir/IR/PatternBase.td"

def ValuesAreEqual : Constraint<CPred<[{ $0 == $1 }]>>;

def ValuesAreNotEqual : Constraint<CPred<[{ $0 != $1 }]>>;

def TypesAreEquivalent
  : Constraint<CPred<[{ clift::equivalent($0.getType(), $1.getType()) }]>>;

def TypesAreNotEquivalent
  : Constraint<CPred<[{ not clift::equivalent($0.getType(), $1.getType()) }]>>;

def IsBitcastKind
  : Constraint<CPred<[{ $0.getValue() == clift::CastKind::Bitcast }]>>;

def getPointeeType : NativeCodeCall<[{
  mlir::cast<clift::PointerType>(mlir::clift::dealias($0.getType(), /*IgnoreQualifiers=*/true)).getPointeeType()
}]>;

def makeInteger : NativeCodeCall<[{
  $_builder.getIntegerAttr($_builder.getIntegerType(64), $0);
}]>;

def forceBooleanResult : NativeCodeCall<[{
  mlir::clift::impl::forceBooleanResult($_builder, $0, $1)
}]>;

#endif
