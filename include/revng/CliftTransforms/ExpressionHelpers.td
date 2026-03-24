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


def getType : NativeCodeCall<[{ $0.getType() }]>;

def getPointeeType : NativeCodeCall<[{
  mlir::cast<clift::PointerType>(mlir::clift::dealias($0.getType(), /*IgnoreQualifiers=*/true)).getPointeeType()
}]>;

def getBitcastKind : NativeCodeCall<[{ clift::CastKind::Bitcast }]>;

def createImmediateOp : NativeCodeCall<[{
  $_builder.create<clift::ImmediateOp>($_loc, $0, $1)
}]>;

def createCastOp : NativeCodeCall<[{
  $_builder.create<clift::CastOp>($_loc, $0, $1, $2)
}]>;

def createAddressofOp : NativeCodeCall<[{
  $_builder.create<clift::AddressofOp>($_loc, $0, $1)
}]>;

def forceBooleanResult : NativeCodeCall<[{
  mlir::clift::impl::forceBooleanResult($_builder, $0, $1)
}]>;

#endif
