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

def getPointeeType : NativeCodeCall<[{
  clift::unwrapped_cast<clift::PointerType>($0.getType()).getPointeeType()
}]>;

def makeInteger : NativeCodeCall<[{
  mlir::IntegerAttr::get($_builder.getContext(), llvm::APSInt($0))
}]>;

def forceBooleanResult : NativeCodeCall<[{
  clift::impl::forceBooleanResult($_builder, $0, $1)
}]>;

#endif
