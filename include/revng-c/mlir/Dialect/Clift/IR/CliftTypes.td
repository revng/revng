//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

// Include the definition of the necessary tablegen constructs for defining our
// types
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SubElementInterfaces.td"
include "Clift.td"
include "CliftAttributes.td"

class Clift_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Clift_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Clift_LabelType : Clift_Type<"Label", "label"> {
  let summary = "Clift label type";

  let description = [{
    Label type.
  }];
}

def Clift_PrimitiveType : Clift_Type<"Primitive", "primitive"> {
  let summary = "Clift primitive type";

  let parameters = (ins  "uint64_t":$ID, "PrimitiveKindAttr":$kind, "uint8_t":$size, "StringAttr":$custom_name, "StringAttr":$original_name);

  let description = [{
    Primitive type.
  }];

  let assemblyFormat = "`<`$ID $kind $size $custom_name $original_name`>` ";
}

