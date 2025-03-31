//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Include the definition of the necessary tablegen constructs for defining our
// types
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SubElementInterfaces.td"

include "revng/mlir/Dialect/Clift/IR/Clift.td"
include "revng/mlir/Dialect/Clift/IR/CliftAttributes.td"
include "revng/mlir/Dialect/Clift/IR/CliftTypeInterfaces.td"

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

def Clift_PrimitiveType
  : Clift_Type<"Primitive",
               "primitive",
               [DeclareTypeInterfaceMethods<Clift_ValueType>,
                DeclareTypeInterfaceMethods<Clift_AliasableType>]> {
  let summary = "Clift primitive type";

  let description = [{
    Primitive type.
  }];

  let parameters = (ins "PrimitiveKind":$kind,
                        "uint64_t":$size,
                        DefaultValuedParameter<"bool", "false">:$is_const);

  let genVerifyDecl = 1;

  let assemblyFormat = [{
    `<` $kind $size `>`
  }];
}

def Clift_PointerType
  : Clift_Type<"Pointer",
               "ptr",
               [SubElementTypeInterface,
                DeclareTypeInterfaceMethods<Clift_ValueType>]> {
  let summary = "Clift pointer type";

  let description = [{
    Pointer type.
  }];

  let parameters = (ins "mlir::clift::ValueType":$pointee_type,
                        "uint64_t":$pointer_size,
                        DefaultValuedParameter<"bool", "false">:$is_const);


  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::clift::ValueType":$PointeeType,
                                        "uint64_t":$PointerSize), [{
      return $_get(PointeeType.getContext(), PointeeType, PointerSize, false);
    }]>,

    TypeBuilderWithInferredContext<(ins "mlir::clift::ValueType":$PointeeType,
                                        "uint64_t":$PointerSize,
                                        "bool":$Const), [{
      return $_get(PointeeType.getContext(), PointeeType, PointerSize, Const);
    }]>,
  ];

  let genVerifyDecl = 1;

  let assemblyFormat = [{
    `<` $pointer_size `to` $pointee_type `>`
  }];

  let extraClassDeclaration = [{
    uint64_t getByteSize() const {
      return getPointerSize();
    }
  }];
}

def Clift_ArrayType
  : Clift_Type<"Array",
               "array",
               [SubElementTypeInterface,
                DeclareTypeInterfaceMethods<Clift_ValueType>]> {
  let summary = "Clift array type";

  let description = [{
    ArrayType type.
  }];

  let parameters = (ins "mlir::clift::ValueType":$element_type,
                        "uint64_t":$elements_count);

  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::clift::ValueType":$ElementType,
                                        "uint64_t":$ElementCount), [{
      return $_get(ElementType.getContext(), ElementType, ElementCount);
    }]>,
  ];

  let genVerifyDecl = 1;

  let assemblyFormat = [{
    `<` $elements_count `x` $element_type `>`
  }];

  let extraClassDeclaration = [{
    bool isConst() const {
      return getElementType().isConst();
    }

    uint64_t getByteSize() const {
      return getElementsCount() * getElementType().getByteSize();
    }
  }];
}

def Clift_EnumType
  : Clift_Type<"Enum",
               "enum",
               [SubElementTypeInterface,
                DeclareTypeInterfaceMethods<Clift_DefinedType>]> {

  let summary = "Clift enum type";
  let description = [{
    Clift enum type.
  }];

  let parameters = (ins "mlir::clift::EnumAttr":$definition,
                        DefaultValuedParameter<"bool", "false">:$is_const);

  let builders = [
    TypeBuilder<(ins "llvm::StringRef":$Handle,
                     "llvm::StringRef":$Name,
                     "mlir::clift::ValueType":$UnderlyingType,
                     "llvm::ArrayRef<mlir::clift::EnumFieldAttr>":$Fields), [{
      return $_get($_ctxt,
                   EnumAttr::get($_ctxt,
                                 Handle,
                                 Name,
                                 UnderlyingType,
                                 Fields),
                   /*IsConst=*/false);
    }]>,

    TypeBuilderWithInferredContext<(ins "llvm::StringRef":$Handle,
                                        "llvm::StringRef":$Name,
                                        "mlir::clift::ValueType":$UnderlyingType,
                                        "llvm::ArrayRef<mlir::clift::EnumFieldAttr>":$Fields), [{
      return $_get(UnderlyingType.getContext(),
                   EnumAttr::get(UnderlyingType.getContext(),
                                 Handle,
                                 Name,
                                 UnderlyingType,
                                 Fields),
                   /*IsConst=*/false);
    }]>,
  ];

  let extraClassDeclaration = [{
    // AliasableType
    bool getAlias(llvm::raw_ostream &OS) const;

    // ValueType
    uint64_t getByteSize() const {
      return getUnderlyingType().getByteSize();
    }

    clift::ValueType addConst() const;
    clift::ValueType removeConst() const;

    llvm::StringRef getHandle() const {
      return getDefinition().getHandle();
    }

    llvm::StringRef getName() const {
      return getDefinition().getName();
    }

    clift::ValueType getUnderlyingType() const {
      return getDefinition().getUnderlyingType();
    }

    llvm::ArrayRef<EnumFieldAttr> getFields() const {
      return getDefinition().getFields();
    }

    using Base::verify;

    static mlir::LogicalResult
    verify(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
           llvm::StringRef Handle,
           llvm::StringRef Name,
           clift::ValueType UnderlyingType,
           llvm::ArrayRef<clift::EnumFieldAttr> Fields) {
      return EnumAttr::verify(EmitError, Handle, Name, UnderlyingType, Fields);
    }
  }];

  let hasCustomAssemblyFormat = 1;
}

def Clift_TypedefType
  : Clift_Type<"Typedef",
               "typedef",
               [SubElementTypeInterface,
                DeclareTypeInterfaceMethods<Clift_DefinedType>]> {

  let summary = "Clift typedef type";
  let description = [{
    Clift typedef type. Behaves like a distinct type rather than an alias for
    the underlying type.
  }];

  let parameters = (ins "mlir::clift::TypedefAttr":$definition,
                        DefaultValuedParameter<"bool", "false">:$is_const);

  let builders = [
    TypeBuilder<(ins "llvm::StringRef":$Handle,
                     "llvm::StringRef":$Name,
                     "mlir::clift::ValueType":$UnderlyingType), [{
      return $_get($_ctxt,
                   TypedefAttr::get($_ctxt,
                                    Handle,
                                    Name,
                                    UnderlyingType),
                   /*IsConst=*/false);
    }]>,

    TypeBuilderWithInferredContext<(ins "llvm::StringRef":$Handle,
                                        "llvm::StringRef":$Name,
                                        "mlir::clift::ValueType":$UnderlyingType), [{
      return TypedefType::get(UnderlyingType.getContext(),
                              Handle,
                              Name,
                              UnderlyingType);
    }]>,
  ];

  let extraClassDeclaration = [{
    // AliasableType
    bool getAlias(llvm::raw_ostream &OS) const;

    // ValueType
    uint64_t getByteSize() const;
    clift::ValueType addConst() const;
    clift::ValueType removeConst() const;

    llvm::StringRef getHandle() const {
      return getDefinition().getHandle();
    }

    llvm::StringRef getName() const {
      return getDefinition().getName();
    }

    clift::ValueType getUnderlyingType() const {
      return getDefinition().getUnderlyingType();
    }

    using Base::verify;

    static mlir::LogicalResult
    verify(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
           llvm::StringRef Handle,
           llvm::StringRef Name,
           clift::ValueType UnderlyingType) {
      return TypedefAttr::verify(EmitError, Handle, Name, UnderlyingType);
    }
  }];

  let hasCustomAssemblyFormat = 1;
}

def Clift_FunctionType
  : Clift_Type<"Function",
               "func",
               [SubElementTypeInterface,
                DeclareTypeInterfaceMethods<Clift_DefinedType>]> {

  let summary = "Clift function type";
  let description = [{
    Clift function type. Unlike C function types, Clift function types have
    strong identity.
  }];

  let parameters = (ins StringRefParameter<>:$handle,
                        StringRefParameter<>:$name,
                        "mlir::Type":$return_type,
                        OptionalArrayRefParameter<"mlir::Type">:$argument_types);

  let builders = [
    TypeBuilder<(ins "llvm::StringRef":$Handle,
                     "llvm::StringRef":$Name,
                     "mlir::clift::ValueType":$ReturnType,
                     "llvm::ArrayRef<mlir::clift::ValueType>":$ArgumentTypes), [{
      return $_get($_ctxt,
                   Handle,
                   Name,
                   ReturnType,
                   llvm::SmallVector<mlir::Type, 16>(ArgumentTypes));
    }]>,

    TypeBuilderWithInferredContext<(ins "llvm::StringRef":$Handle,
                                        "llvm::StringRef":$Name,
                                        "mlir::Type":$ReturnType,
                                        "llvm::ArrayRef<mlir::Type>":$ArgumentTypes), [{
      return $_get(ReturnType.getContext(),
                   Handle,
                   Name,
                   ReturnType,
                   ArgumentTypes);
    }]>,

    TypeBuilderWithInferredContext<(ins "llvm::StringRef":$Handle,
                                        "llvm::StringRef":$Name,
                                        "mlir::clift::ValueType":$ReturnType,
                                        "llvm::ArrayRef<mlir::clift::ValueType>":$ArgumentTypes), [{
      return $_get(ReturnType.getContext(),
                   Handle,
                   Name,
                   ReturnType,
                   llvm::SmallVector<mlir::Type, 16>(ArgumentTypes));
    }]>,
  ];

  let extraClassDeclaration = [{
    // AliasableType
    bool getAlias(llvm::raw_ostream &OS) const;

    // ValueType
    uint64_t getByteSize() const {
      return 0;
    }

    clift::ValueType addConst() const {
      return *this;
    }

    clift::ValueType removeConst() const {
      return *this;
    }

    bool isConst() const {
      return false;
    }

    llvm::ArrayRef<mlir::Type> getResultTypes() const;

    static mlir::LogicalResult
    verify(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
           llvm::StringRef Handle,
           llvm::StringRef Name,
           clift::ValueType ReturnType,
           llvm::ArrayRef<clift::ValueType> ArgumentTypes);
  }];

  let genVerifyDecl = 1;

  let assemblyFormat = [{
    `<`
    $handle custom<CliftDebugName>($name)
    `:` $return_type `(` (`)`) : ($argument_types^ `)`)?
    `>`
  }];
}

class Clift_ClassTypeBase<string name, string mnemonic>
  : Clift_Type<name,
               mnemonic,
               [SubElementTypeInterface,
                DeclareTypeInterfaceMethods<Clift_ClassType>]> {

  let summary = "Clift " # mnemonic # " type";

  let parameters = (ins "mlir::clift::" # name # "Attr":$definition,
                        DefaultValuedParameter<"bool", "false">:$is_const);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    using DefinitionAttrType = }] # name # [{Attr;

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

    // AliasableType
    bool getAlias(llvm::raw_ostream &OS) const;

    // ValueType
    clift::ValueType addConst() const;
    clift::ValueType removeConst() const;


    using Base::verify;

    template<typename... ArgsT>
    static mlir::LogicalResult
    verify(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
           llvm::StringRef Handle,
           const ArgsT &... Args) {
      return DefinitionAttrType::verify(EmitError, Handle, Args...);
    }
  }];

  let extraClassDefinition = [{
    clift::ValueType $cppClass::addConst() const {
      return Base::get(getContext(), getDefinition(), true);
    }

    clift::ValueType $cppClass::removeConst() const {
      return Base::get(getContext(), getDefinition(), false);
    }
  }];
}

def Clift_StructType : Clift_ClassTypeBase<"Struct", "struct"> {
  let builders = [
    TypeBuilder<(ins "llvm::StringRef":$Handle), [{
      return $_get($_ctxt, StructAttr::get($_ctxt, Handle), /*IsConst=*/false);
    }]>,

    TypeBuilder<(ins "llvm::StringRef":$Handle,
                     "llvm::StringRef":$Name,
                     "uint64_t":$Size,
                     "llvm::ArrayRef<mlir::clift::FieldAttr>":$Fields), [{
      return $_get($_ctxt,
                   StructAttr::get($_ctxt,
                                   Handle,
                                   Name,
                                   Size,
                                   Fields),
                   /*IsConst=*/false);
    }]>,
  ];
}

def Clift_UnionType : Clift_ClassTypeBase<"Union", "union"> {
  let builders = [
    TypeBuilder<(ins "llvm::StringRef":$Handle), [{
      return $_get($_ctxt, UnionAttr::get($_ctxt, Handle), /*IsConst=*/false);
    }]>,

    TypeBuilder<(ins "llvm::StringRef":$Handle,
                     "llvm::StringRef":$Name,
                     "llvm::ArrayRef<mlir::clift::FieldAttr>":$Fields), [{
      return $_get($_ctxt,
                   UnionAttr::get($_ctxt,
                                  Handle,
                                  Name,
                                  Fields),
                   /*IsConst=*/false);
    }]>,
  ];
}
