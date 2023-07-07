//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def PrimitiveKind: I32EnumAttr<"PrimitiveKind", "primitive type kinds", [
    I32EnumAttrCase<"Void", 0>,
    I32EnumAttrCase<"Generic", 1>,
    I32EnumAttrCase<"PointerOrNumber", 2>,
    I32EnumAttrCase<"Number", 3>,
    I32EnumAttrCase<"Unsigned", 4>,
    I32EnumAttrCase<"Signed", 5>,
    I32EnumAttrCase<"Float", 6>,
  ]> {
  let cppNamespace = "::mlir::clift";
}
