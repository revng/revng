//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_ATTRIBUTE
#define MLIR_CLIFT_ATTRIBUTE
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SubElementInterfaces.td"

include "revng/mlir/Dialect/Clift/IR/Clift.td"
include "revng/mlir/Dialect/Clift/IR/CliftAttrInterfaces.td"
include "revng/mlir/Dialect/Clift/IR/CliftEnums.td"

class Clift_Attr<string name, string attrMnemonic, list<Trait> traits = []>
  : AttrDef<Clift_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Clift_FieldAttr
  : AttrDef<Clift_Dialect,
            "Field",
            [SubElementAttrInterface]> {

  let summary = "A attribute representing a field of a struct or a union";
  let description = [{
    A attribute representing a field of a struct or a union
  }];

  let parameters = (ins "uint64_t":$offset,
                        "mlir::clift::ValueType":$type,
                        StringRefParameter<>:$name);

  let builders = [
    AttrBuilderWithInferredContext<(ins "uint64_t":$Offset,
                                        "mlir::clift::ValueType":$Type,
                                        CArg<"llvm::StringRef", "\"\"">:$Name), [{
      return $_get(Type.getContext(), Offset, Type, Name);
    }]>,

    AttrBuilderWithInferredContext<(ins "mlir::clift::ValueType":$Type,
                                        CArg<"llvm::StringRef", "\"\"">:$Name), [{
      return $_get(Type.getContext(), 0, Type, Name);
    }]>
  ];

  let genVerifyDecl = 1;
}

def Clift_EnumFieldAttr
  : AttrDef<Clift_Dialect,
            "EnumField",
            [SubElementAttrInterface]> {

  let summary = "A attribute representing a field of enum";
  let description = [{
    A attribute representing a field of a struct or a union
  }];

  let parameters = (ins "uint64_t":$raw_value,
                        StringRefParameter<>:$name);

  let genVerifyDecl = 1;
}

def Clift_EnumAttr
  : AttrDef<Clift_Dialect,
            "Enum",
            [Clift_TypeDefinitionAttr,
             SubElementAttrInterface]> {

  let parameters = (ins StringRefParameter<>:$handle,
                        StringRefParameter<>:$name,
                        "mlir::clift::ValueType":$underlying_type,
                        ArrayRefParameter<"mlir::clift::EnumFieldAttr">:$fields);

  let genVerifyDecl = 1;
}

def Clift_TypedefAttr
  : AttrDef<Clift_Dialect,
            "Typedef",
            [Clift_TypeDefinitionAttr,
             SubElementAttrInterface]> {

  let parameters = (ins StringRefParameter<>:$handle,
                        StringRefParameter<>:$name,
                        "mlir::clift::ValueType":$underlying_type);

  let genVerifyDecl = 1;
}

// This is a helper attribute used for emitting the bytecode representation of
// (potentially recursive) class types.
def Clift_BytecodeClassAttr
  : AttrDef<Clift_Dialect,
            "BytecodeClass",
            [SubElementAttrInterface]> {

  let parameters = (ins "mlir::clift::ClassType":$type);
}

#endif
