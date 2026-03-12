//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Include the definition of the necessary tablegen constructs for defining our
// types
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SubElementInterfaces.td"

include "revng/Clift/CliftDialect.td"
include "revng/Clift/CliftAttributes.td"
include "revng/Clift/CliftTypeInterfaces.td"

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

def Clift_VoidType
  : Clift_Type<"Void",
               "void",
               [Clift_PrimitiveType,
                DeclareTypeInterfaceMethods<Clift_ValueType, ["getByteSize"]>,
                DeclareTypeInterfaceMethods<Clift_AliasableType>]> {
  let summary = "Clift void type";

  let parameters = (ins DefaultValuedParameter<"bool", "false">:$is_const);

  let builders = [
    TypeBuilder<(ins), [{
      return $_get($_ctxt, false);
    }]>,
  ];

  let assemblyFormat = [{}];

  let genVerifyDecl = 1;
}

def Clift_IntegerType
  : Clift_Type<"Integer",
               "int",
               [Clift_PrimitiveType,
                DeclareTypeInterfaceMethods<Clift_ValueType>,
                DeclareTypeInterfaceMethods<Clift_AliasableType>]> {
  let summary = "Clift integer type";

  let parameters = (ins "IntegerKind":$kind,
                        "uint64_t":$size,
                        DefaultValuedParameter<"bool", "false">:$is_const);

  let builders = [
    TypeBuilder<(ins "IntegerKind":$Kind, "uint64_t":$Size), [{
      return $_get($_ctxt, Kind, Size, false);
    }]>,
  ];

  let assemblyFormat = [{
    `<` $kind $size `>`
  }];

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    bool isSigned() { return getKind() == IntegerKind::Signed; }
  }];
}

def Clift_FloatType
  : Clift_Type<"Float",
               "float",
               [Clift_PrimitiveType,
                DeclareTypeInterfaceMethods<Clift_ValueType>,
                DeclareTypeInterfaceMethods<Clift_AliasableType>]> {
  let summary = "Clift float type";

  let parameters = (ins "uint64_t":$size,
                        DefaultValuedParameter<"bool", "false">:$is_const);

  let builders = [
    TypeBuilder<(ins "uint64_t":$Size), [{
      return $_get($_ctxt, Size, false);
    }]>,
  ];

  let assemblyFormat = [{
    `<` $size `>`
  }];

  let genVerifyDecl = 1;
}

def Clift_PointerType
  : Clift_Type<"Pointer",
               "ptr",
               [SubElementTypeInterface,
                DeclareTypeInterfaceMethods<Clift_ValueType, ["getByteSize"]>]> {
  let summary = "Clift pointer type";

  let description = [{
    Pointer type.
  }];

  let parameters = (ins "mlir::Type":$pointee_type,
                        "uint64_t":$pointer_size,
                        DefaultValuedParameter<"bool", "false">:$is_const);

  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::Type":$PointeeType,
                                        "uint64_t":$PointerSize,
                                        CArg<"bool", "false">:$Const), [{
      return $_get(PointeeType.getContext(), PointeeType, PointerSize, Const);
    }]>,
  ];

  let genVerifyDecl = 1;

  let assemblyFormat = [{
    `<` $pointer_size `to` $pointee_type `>`
  }];
}

def Clift_ArrayType
  : Clift_Type<"Array",
               "array",
               [SubElementTypeInterface,
                DeclareTypeInterfaceMethods<Clift_ValueType, ["getByteSize", "isConst"]>]> {
  let summary = "Clift array type";

  let description = [{
    ArrayType type.
  }];

  let parameters = (ins "mlir::Type":$element_type,
                        "uint64_t":$elements_count);

  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::Type":$ElementType,
                                        "uint64_t":$ElementCount), [{
      return $_get(ElementType.getContext(), ElementType, ElementCount);
    }]>,
  ];

  let genVerifyDecl = 1;

  let assemblyFormat = [{
    `<` $elements_count `x` $element_type `>`
  }];
}

def Clift_EnumType
  : Clift_Type<"Enum",
               "enum",
               [SubElementTypeInterface,
                DeclareTypeInterfaceMethods<Clift_AliasableType>,
                DeclareTypeInterfaceMethods<Clift_ValueType, ["getByteSize"]>,
                DeclareTypeInterfaceMethods<Clift_DefinedType>]> {

  let summary = "Clift enum type";
  let description = [{
    Clift enum type.
  }];

  let parameters = (ins "mlir::clift::EnumAttr":$definition,
                        DefaultValuedParameter<"bool", "false">:$is_const);

  let builders = [
    TypeBuilder<(ins "mlir::clift::EnumAttr":$Definition), [{
      return $_get($_ctxt, Definition, /*IsConst=*/false);
    }]>,

    TypeBuilderWithInferredContext<(ins "mlir::clift::EnumAttr":$Definition), [{
      return $_get(Definition.getContext(), Definition, /*IsConst=*/false);
    }]>,
  ];

  let extraClassDeclaration = [{
    EnumFieldAttr getFieldByValue(uint64_t RawValue) const {
      for (EnumFieldAttr Field : getFields()) {
        if (Field.getRawValue() == RawValue)
          return Field;
      }
      return {};
    }

    llvm::StringRef getHandle() const {
      return getDefinition().getHandle();
    }

    llvm::StringRef getName() const {
      return getDefinition().getName();
    }

    mlir::Type getUnderlyingType() const {
      return getDefinition().getUnderlyingType();
    }

    llvm::ArrayRef<EnumFieldAttr> getFields() const {
      return getDefinition().getFields();
    }
  }];

  let hasCustomAssemblyFormat = 1;
}

def Clift_TypedefType
  : Clift_Type<"Typedef",
               "typedef",
               [SubElementTypeInterface,
                DeclareTypeInterfaceMethods<Clift_AliasableType>,
                DeclareTypeInterfaceMethods<Clift_ValueType, ["getByteSize"]>,
                DeclareTypeInterfaceMethods<Clift_DefinedType>]> {

  let summary = "Clift typedef type";
  let description = [{
    Clift typedef type. Behaves like a distinct type rather than an alias for
    the underlying type.
  }];

  let parameters = (ins "mlir::clift::TypedefAttr":$definition,
                        DefaultValuedParameter<"bool", "false">:$is_const);

  let builders = [
    TypeBuilder<(ins "mlir::clift::TypedefAttr":$Definition), [{
      return $_get($_ctxt, Definition, /*IsConst=*/false);
    }]>,

    TypeBuilderWithInferredContext<(ins "mlir::clift::TypedefAttr":$Definition), [{
      return $_get(Definition.getContext(), Definition, /*IsConst=*/false);
    }]>,
  ];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef NameAttrKey = TypedefAttr::NameAttrKey;
    static constexpr llvm::StringRef CommentAttrKey = TypedefAttr::CommentAttrKey;

    llvm::StringRef getHandle() const {
      return getDefinition().getHandle();
    }

    llvm::StringRef getName() const {
      return getDefinition().getName();
    }

    mlir::Type getUnderlyingType() const {
      return getDefinition().getUnderlyingType();
    }
  }];

  let hasCustomAssemblyFormat = 1;
}

// TODO: Investigate whether there is enough value in unifying model and Clift
//       function types to justify doing so, and how that should be done.
def Clift_FunctionType
  : Clift_Type<"Function",
               "func",
               [SubElementTypeInterface,
                DeclareTypeInterfaceMethods<Clift_AliasableType>,
                DeclareTypeInterfaceMethods<Clift_ValueType, ["getByteSize", "isConst"]>,
                DeclareTypeInterfaceMethods<Clift_DefinedType>]> {

  let summary = "Clift function type";
  let description = [{
    Clift function type. Unlike C function types, Clift function types have
    strong identity.
  }];

  let parameters = (ins StringRefParameter<>:$handle,
                        "mlir::clift::MutableStringAttr":$mutable_name,
                        "mlir::clift::MutableStringAttr":$mutable_comment,
                        "mlir::Type":$return_type,
                        OptionalArrayRefParameter<"mlir::Type">:$argument_types,
                        OptionalArrayRefParameter<"mlir::clift::CAttributeAttr">:$c_attributes);

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef NameAttrKey = "function-name";
    static constexpr llvm::StringRef CommentAttrKey = "function-comment";

    llvm::StringRef getName() const {
      return getMutableName().getValue();
    }

    llvm::StringRef getComment() const {
      return getMutableComment().getValue();
    }

    llvm::ArrayRef<mlir::Type> getResultTypes() const;
  }];

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

class Clift_ClassTypeBase<string name, string mnemonic>
  : Clift_Type<name,
               mnemonic,
               [SubElementTypeInterface,
                DeclareTypeInterfaceMethods<Clift_AliasableType>,
                DeclareTypeInterfaceMethods<Clift_ValueType>,
                DeclareTypeInterfaceMethods<Clift_ClassType>]> {

  let summary = "Clift " # mnemonic # " type";

  let parameters = (ins "mlir::clift::" # name # "Attr":$definition,
                        DefaultValuedParameter<"bool", "false">:$is_const);

  let extraClassDeclaration = [{
    using DefinitionAttrType = }] # name # [{Attr;

    static constexpr llvm::StringRef NameAttrKey = DefinitionAttrType::NameAttrKey;
    static constexpr llvm::StringRef CommentAttrKey = DefinitionAttrType::CommentAttrKey;

    bool isComplete() const {
      return getDefinition().hasDefinition();
    }

    llvm::StringRef getHandle() const {
      return getDefinition().getHandle();
    }

    llvm::StringRef getName() const {
      return getDefinition().getName();
    }

    uint64_t getSize() const;

    llvm::ArrayRef<clift::FieldAttr> getFields() const {
      return getDefinition().getFields();
    }

    llvm::ArrayRef<mlir::clift::CAttributeAttr> getCAttributes() const {
      return getDefinition().getCAttributes();
    }
  }];

  let extraClassDefinition = [{
    clift::ValueType $cppClass::addConst() const {
      return Base::get(getContext(), getDefinition(), /*IsConst=*/true);
    }

    clift::ValueType $cppClass::removeConst() const {
      return Base::get(getContext(), getDefinition(), /*IsConst=*/false);
    }
  }];

  let hasCustomAssemblyFormat = 1;
}

def Clift_StructType : Clift_ClassTypeBase<"Struct", "struct"> {
  let builders = [
    TypeBuilder<(ins "llvm::StringRef":$Handle,
                     CArg<"bool", "false">:$IsConst), [{
      return $_get($_ctxt, StructAttr::get($_ctxt, Handle), IsConst);
    }]>,

    TypeBuilder<(ins "mlir::clift::StructAttr":$Definition), [{
      return $_get($_ctxt, Definition, /*IsConst=*/false);
    }]>,

    TypeBuilderWithInferredContext<(ins "mlir::clift::StructAttr":$Definition,
                                        CArg<"bool", "false">:$IsConst), [{
      return $_get(Definition.getContext(), Definition, IsConst);
    }]>,
  ];
}

def Clift_UnionType : Clift_ClassTypeBase<"Union", "union"> {
  let builders = [
    TypeBuilder<(ins "llvm::StringRef":$Handle,
                     CArg<"bool", "false">:$IsConst), [{
      return $_get($_ctxt, UnionAttr::get($_ctxt, Handle), IsConst);
    }]>,

    TypeBuilder<(ins "mlir::clift::UnionAttr":$Definition), [{
      return $_get($_ctxt, Definition, /*IsConst=*/false);
    }]>,

    TypeBuilderWithInferredContext<(ins "mlir::clift::UnionAttr":$Definition,
                                        CArg<"bool", "false">:$IsConst), [{
      return $_get(Definition.getContext(), Definition, IsConst);
    }]>,
  ];
}
