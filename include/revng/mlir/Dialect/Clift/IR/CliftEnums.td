//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

include "revng/mlir/Dialect/Clift/IR/Clift.td"

def Clift_PrimitiveKind: I32EnumAttr<"PrimitiveKind", "primitive type kinds", [
    I32EnumAttrCase<"VoidKind",                 1,      "void">,
    I32EnumAttrCase<"GenericKind",              2,      "generic">,
    I32EnumAttrCase<"PointerOrNumberKind",      3,      "pointer_or_number">,
    I32EnumAttrCase<"NumberKind",               4,      "number">,
    I32EnumAttrCase<"UnsignedKind",             5,      "unsigned">,
    I32EnumAttrCase<"SignedKind",               6,      "signed">,
    I32EnumAttrCase<"FloatKind",                7,      "float">,
  ]> {
  let cppNamespace = "mlir::clift";
  // since we want to wrap this enum into a clift type, do not generate
  // attribute wrapping it.
  let genSpecializedAttr = 0;
}

def Clift_CastKind : I32EnumAttr<"CastKind", "Cast operation semantic kinds", [
    I32EnumAttrCase<"Extend",           1,      "extend">,
    I32EnumAttrCase<"Truncate",         2,      "truncate">,
    I32EnumAttrCase<"Bitcast",          3,      "bitcast">,
    I32EnumAttrCase<"Decay",            4,      "decay">,
    I32EnumAttrCase<"Convert",          5,      "convert">,
  ]> {
  let cppNamespace = "mlir::clift";
}
