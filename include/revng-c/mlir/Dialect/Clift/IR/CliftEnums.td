//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "Clift.td"

def PrimitiveKind: I32EnumAttr<"PrimitiveKind", "primitive type kinds", [
    I32EnumAttrCase<"VoidKind", 1>,
    I32EnumAttrCase<"GenericKind", 2>,
    I32EnumAttrCase<"PointerOrNumberKind", 3>,
    I32EnumAttrCase<"NumberKind", 4>,
    I32EnumAttrCase<"UnsignedKind", 5>,
    I32EnumAttrCase<"SignedKind", 6>,
    I32EnumAttrCase<"FloatKind", 7>,
  ]> {
  let cppNamespace = "::mlir::clift";
  // since we want to wrap this enum into a clift type, do not generate
  // attribute wrapping it.
  let genSpecializedAttr = 0;
}

