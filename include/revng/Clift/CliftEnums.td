//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

include "revng/Clift/CliftDialect.td"

def Clift_IntegerKind: I32EnumAttr<"IntegerKind", "integer type kinds", [
    I32EnumAttrCase<"Generic",                  1,      "generic">,
    I32EnumAttrCase<"PointerOrNumber",          2,      "pointer_or_number">,
    I32EnumAttrCase<"Number",                   3,      "number">,
    I32EnumAttrCase<"Unsigned",                 4,      "unsigned">,
    I32EnumAttrCase<"Signed",                   5,      "signed">,
  ]> {
  let cppNamespace = "mlir::clift";
  // since we want to wrap this enum into a clift type, do not generate
  // attribute wrapping it.
  let genSpecializedAttr = 0;
}
