//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_ATTRIBUTE
#define MLIR_CLIFT_ATTRIBUTE
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SubElementInterfaces.td"
include "Clift.td"
include "CliftEnums.td"
include "CliftInterfaces.td"

class Clift_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Clift_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def FunctionArgumentAttr: Clift_Attr<"FunctionArgument", "farg", [DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
  let summary = "A attribute representing a argument of a function";
  let description = [{
    A attribute representing a argument of a function
  }];

  let parameters = (ins "mlir::clift::ValueType":$type, StringRefParameter<>:$name);

  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::Type":$Type,
                                        CArg<"llvm::StringRef", "\"\"">:$Name), [{
      return $_get(Type.getContext(), Type, Name);
    }]>
  ];

  let genVerifyDecl = 1;

  let assemblyFormat = "`<`struct($name, $type)`>`";
}

def FieldAttr : Clift_Attr<"Field", "field", [DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
  let summary = "A attribute representing a field of a struct or a union";
  let description = [{
    A attribute representing a field of a struct or a union
  }];

  let parameters = (ins "uint64_t":$offset, "mlir::Type":$type, StringRefParameter<>:$name);

  let builders = [
    AttrBuilderWithInferredContext<(ins "uint64_t":$Offset,
                                        "mlir::Type":$Type,
                                        CArg<"llvm::StringRef", "\"\"">:$Name), [{
      return $_get(Type.getContext(), Offset, Type, Name);
    }]>,
    AttrBuilderWithInferredContext<(ins "mlir::Type":$Type,
                                        CArg<"llvm::StringRef", "\"\"">:$Name), [{
      return $_get(Type.getContext(), 0, Type, Name);
    }]>
  ];

  let genVerifyDecl = 1;

  let assemblyFormat = "`<`struct($offset, $name, $type)`>`";
}

def EnumFieldAttr : Clift_Attr<"EnumField", "enum_field", [DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {
  let summary = "A attribute representing a field of enum";
  let description = [{
    A attribute representing a field of a struct or a union
  }];

  let parameters = (ins "uint64_t":$raw_value, StringRefParameter<>:$name);

  let assemblyFormat = "`<`struct($name, $raw_value)`>`";
}

def Clift_EnumAttr : Clift_Attr<"Enum", "enum", [DeclareAttrInterfaceMethods<AliasableAttr>, DeclareAttrInterfaceMethods<SizedType, ["getByteSize"]>, TypeDefinition, SubElementAttrInterface]> {
  let summary = "Clift enum type";

  let parameters = (ins "uint64_t":$id, StringRefParameter<>:$name, "mlir::Type":$underlying_type, ArrayRefParameter<"mlir::clift::EnumFieldAttr">:$fields);

  let builders = [
    TypeBuilderWithInferredContext<(ins "uint64_t":$ID, "llvm::StringRef":$Name, "mlir::Type":$ElementType, "llvm::ArrayRef<mlir::clift::EnumFieldAttr>":$Fields), [{
      return EnumAttr::get(ElementType.getContext(), ID, Name, ElementType, Fields);
     }]>,
  ];

  let description = [{
    Enum type.
  }];
  let genVerifyDecl = 1;

  let assemblyFormat = "`<`struct($id, $name, $underlying_type)  `,` `fields` `=` `[` $fields `]``>` ";
}

def Clift_TypedefAttr: Clift_Attr<"Typedef", "typedef", [DeclareAttrInterfaceMethods<AliasableAttr>, DeclareAttrInterfaceMethods<SizedType, ["getByteSize"]>, TypeDefinition, SubElementAttrInterface]> {
  let summary = "Clift typedef type";

  let parameters = (ins "uint64_t":$id, StringRefParameter<>:$name, "mlir::clift::ValueType":$underlying_type);

  let builders = [
    TypeBuilderWithInferredContext<(ins "uint64_t":$ID, "llvm::StringRef":$Name, "mlir::clift::ValueType":$ElementType), [{
      return TypedefAttr::get(ElementType.getContext(), ID, Name, ElementType);
     }]>,
  ];

  let description = [{
    Typedef type.
  }];

  let assemblyFormat = "`<`struct($id, $name, $underlying_type) `>` ";
}

def Clift_FunctionAttr : Clift_Attr<"Function", "function", [DeclareAttrInterfaceMethods<AliasableAttr>, DeclareAttrInterfaceMethods<SizedType, ["getByteSize"]>, TypeDefinition, SubElementAttrInterface]> {
  let summary = "Clift function type";

  let parameters = (ins "uint64_t":$id, StringRefParameter<>:$name, "mlir::clift::ValueType":$return_type, ArrayRefParameter<"mlir::clift::FunctionArgumentAttr">:$argument_types);

  let builders = [
    TypeBuilderWithInferredContext<(ins "uint64_t":$ID, "mlir::clift::ValueType":$ReturnType,   CArg<"llvm::ArrayRef<mlir::clift::FunctionArgumentAttr>", "{}">:$InputTypes, CArg<"llvm::StringRef", "\"\"">:$Name), [{
      return FunctionAttr::get(ReturnType.getContext(), ID, Name, ReturnType, InputTypes);
     }]>,
  ];

  let description = [{
    TypeDef type.
  }];
  let genVerifyDecl = 1;

  let assemblyFormat = "`<`struct($id, $name, $return_type) `,` `argument_types` `=` `[` $argument_types `]``>` ";
}

#endif
