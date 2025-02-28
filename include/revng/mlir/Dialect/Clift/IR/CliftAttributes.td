//
// This file is distributed under the MIT License. See LICENSE.md for details.
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

def Clift_FieldAttr
  : Clift_Attr<"Field",
               "field",
               [DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {

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

  let assemblyFormat = "`<`struct($offset, $name, $type)`>`";
}

def Clift_EnumFieldAttr
  : AttrDef<Clift_Dialect,
            "EnumField",
            [DeclareAttrInterfaceMethods<SubElementAttrInterface>]> {

  let summary = "A attribute representing a field of enum";
  let description = [{
    A attribute representing a field of a struct or a union
  }];

  let parameters = (ins "uint64_t":$raw_value,
                        StringRefParameter<>:$name);

  let genVerifyDecl = 1;
}

def Clift_EnumTypeAttr
  : Clift_Attr<"EnumType",
               "enum",
               [DeclareAttrInterfaceMethods<Clift_AliasableAttr>,
                DeclareAttrInterfaceMethods<Clift_TypeDefinitionAttr,
                                            ["getByteSize"]>,
                SubElementAttrInterface]> {

  let summary = "Clift enum type";

  let parameters = (ins StringRefParameter<>:$unique_handle,
                        StringRefParameter<>:$name,
                        "mlir::clift::ValueType":$underlying_type,
                        ArrayRefParameter<"mlir::clift::EnumFieldAttr">:$fields);

  let builders = [
    TypeBuilderWithInferredContext<(ins "llvm::StringRef":$UniqueHandle,
                                        "llvm::StringRef":$Name,
                                        "mlir::clift::ValueType":$ElementType,
                                        "llvm::ArrayRef<mlir::clift::EnumFieldAttr>":$Fields), [{
      return EnumTypeAttr::get(ElementType.getContext(),
                               UniqueHandle,
                               Name,
                               ElementType,
                               Fields);
    }]>,
  ];

  let description = [{
    Enum type.
  }];
  let genVerifyDecl = 1;

  let assemblyFormat = [{
    `<`
    $unique_handle custom<CliftDebugName>($name) `:` $underlying_type
    custom<CliftEnumerators>($fields)
    `>`
  }];
}

def Clift_TypedefTypeAttr
  : Clift_Attr<"TypedefType",
               "typedef",
               [DeclareAttrInterfaceMethods<Clift_AliasableAttr>,
                DeclareAttrInterfaceMethods<Clift_TypeDefinitionAttr,
                                            ["getByteSize"]>,
                SubElementAttrInterface]> {

  let summary = "Clift typedef type";

  let parameters = (ins StringRefParameter<>:$unique_handle,
                        StringRefParameter<>:$name,
                        "mlir::clift::ValueType":$underlying_type);

  let builders = [
    TypeBuilderWithInferredContext<(ins "llvm::StringRef":$UniqueHandle,
                                        "llvm::StringRef":$Name,
                                        "mlir::clift::ValueType":$ElementType), [{
      return TypedefTypeAttr::get(ElementType.getContext(),
                                  UniqueHandle,
                                  Name,
                                  ElementType);
    }]>,
  ];

  let description = [{
    Typedef type.
  }];
  let genVerifyDecl = 1;

  let assemblyFormat = [{
    `<` $unique_handle custom<CliftDebugName>($name) `:` $underlying_type `>`
  }];
}

def Clift_FunctionTypeAttr
  : Clift_Attr<"FunctionType",
               "function",
               [DeclareAttrInterfaceMethods<Clift_AliasableAttr>,
                DeclareAttrInterfaceMethods<Clift_TypeDefinitionAttr,
                                            ["getByteSize"]>,
                SubElementAttrInterface]> {

  let summary = "Clift function type";

  let parameters = (ins StringRefParameter<>:$unique_handle,
                        StringRefParameter<>:$name,
                        "mlir::Type":$return_type,
                        OptionalArrayRefParameter<"mlir::Type">:$argument_types);

  let builders = [
    TypeBuilder<(ins "llvm::StringRef":$UniqueHandle,
                     "llvm::StringRef":$Name,
                     "mlir::clift::ValueType":$ReturnType,
                     "llvm::ArrayRef<mlir::clift::ValueType>":$ArgumentTypes), [{
      return $_get($_ctxt,
                   UniqueHandle,
                   Name,
                   ReturnType,
                   llvm::SmallVector<mlir::Type, 16>(ArgumentTypes));
    }]>,

    TypeBuilderWithInferredContext<(ins "llvm::StringRef":$UniqueHandle,
                                        "llvm::StringRef":$Name,
                                        "mlir::clift::ValueType":$ReturnType,
                                        "llvm::ArrayRef<mlir::clift::ValueType>":$ArgumentTypes), [{
      return $_get(ReturnType.getContext(),
                   UniqueHandle,
                   Name,
                   ReturnType,
                   llvm::SmallVector<mlir::Type, 16>(ArgumentTypes));
    }]>,
  ];

  let extraClassDeclaration = [{
    llvm::ArrayRef<mlir::Type> getResultTypes();

    mlir::LogicalResult verify(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
                               llvm::StringRef UniqueHandle,
                               llvm::StringRef Name,
                               clift::ValueType ReturnType,
                               llvm::ArrayRef<clift::ValueType> ArgumentTypes);
  }];

  let description = [{
    TypeDef type.
  }];
  let genVerifyDecl = 1;

  let assemblyFormat = [{
    `<` struct($unique_handle, $name, $return_type) `,` `argument_types` `=` `[` (`]`) : ($argument_types^ `]`)?`>`
  }];
}

#endif
