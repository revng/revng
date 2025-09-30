//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_ATTRIBUTE
#define MLIR_CLIFT_ATTRIBUTE
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SubElementInterfaces.td"

include "revng/mlir/Dialect/Clift/IR/CliftDialect.td"
include "revng/mlir/Dialect/Clift/IR/CliftAttrInterfaces.td"
include "revng/mlir/Dialect/Clift/IR/CliftEnums.td"

class Clift_OptionalArrayRefParameter<string elementType, string desc = "">
  : AttrOrTypeParameter<"std::optional<llvm::ArrayRef<" # elementType # ">>", desc> {

  // Describes how to make a copy of the string, backed by the attribute's
  // storage allocator. $_dst is the std::optional<llvm::ArrayRef> stored in the
  // attribute implementation object. $_self is the argument, which may dangle
  // after this function returns if a reference to a temporary array was given.
  let allocator = [{
    if (auto const& self = $_self) {
      $_dst = $_allocator.copyInto(*self);
    }
  }];
}

class Clift_Attr<string name, string attrMnemonic, list<Trait> traits = []>
  : AttrDef<Clift_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Clift_StringPairAttr : AttrDef<Clift_Dialect, "StringPair"> {
  let parameters = (ins StringRefParameter<>:$first,
                        StringRefParameter<>:$second);
}

def Clift_AttributeComponentAttr : AttrDef<Clift_Dialect,
                                           "AttributeComponent",
                                           [SubElementAttrInterface]> {
  let parameters = (ins StringRefParameter<>:$string,
                        StringRefParameter<>:$handle);
}

def Clift_AttributeAttr : Clift_Attr<"Attribute",
                                     "attribute",
                                     [SubElementAttrInterface]> {
  let parameters = (ins "mlir::clift::AttributeComponentAttr":$macro,
                        Clift_OptionalArrayRefParameter<"mlir::clift::AttributeComponentAttr">:$arguments);

  let hasCustomAssemblyFormat = 1;
}

def Clift_FieldAttr
  : AttrDef<Clift_Dialect,
            "Field",
            [SubElementAttrInterface]> {

  let summary = "A attribute representing a field of a struct or a union";
  let description = [{
    A attribute representing a field of a struct or a union
  }];

  let parameters = (ins StringRefParameter<>:$handle,
                        "mlir::clift::MutableStringAttr":$mutable_name,
                        "uint64_t":$offset,
                        "mlir::clift::ValueType":$type);

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef NameAttrKey = "Field";

    llvm::StringRef getName() const {
      return getMutableName().getValue();
    }
  }];

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

  let parameters = (ins StringRefParameter<>:$handle,
                        "mlir::clift::MutableStringAttr":$mutable_name,
                        "uint64_t":$raw_value);

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef NameAttrKey = "EnumField";

    llvm::StringRef getName() const {
      return getMutableName().getValue();
    }
  }];

  let genVerifyDecl = 1;
}

def Clift_EnumAttr
  : AttrDef<Clift_Dialect,
            "Enum",
            [Clift_TypeDefinitionAttr,
             SubElementAttrInterface]> {

  let parameters = (ins StringRefParameter<>:$handle,
                        "mlir::clift::MutableStringAttr":$mutable_name,
                        "mlir::clift::ValueType":$underlying_type,
                        ArrayRefParameter<"mlir::clift::EnumFieldAttr">:$fields);

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef NameAttrKey = "Enum";

    llvm::StringRef getName() const {
      return getMutableName().getValue();
    }
  }];

  let genVerifyDecl = 1;
}

def Clift_TypedefAttr
  : AttrDef<Clift_Dialect,
            "Typedef",
            [Clift_TypeDefinitionAttr,
             SubElementAttrInterface]> {

  let parameters = (ins StringRefParameter<>:$handle,
                        "mlir::clift::MutableStringAttr":$mutable_name,
                        "mlir::clift::ValueType":$underlying_type);

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef NameAttrKey = "Typedef";

    llvm::StringRef getName() const {
      return getMutableName().getValue();
    }
  }];

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
