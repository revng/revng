//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_ATTRIBUTE
#define MLIR_CLIFT_ATTRIBUTE

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SubElementInterfaces.td"

include "revng/Clift/CliftDialect.td"
include "revng/Clift/CliftAttrInterfaces.td"
include "revng/Clift/CliftEnums.td"

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

def Clift_CIdentifierAttr : Clift_Attr<"CIdentifier",
                                             "identifier",
                                             [SubElementAttrInterface]> {
  let parameters = (ins StringRefParameter<>:$name,
                        StringRefParameter<>:$handle);

  let hasCustomAssemblyFormat = 1;
}

def Clift_CAttributeAttr : Clift_Attr<"CAttribute",
                                      "c_attribute",
                                      [SubElementAttrInterface]> {
  let parameters = (ins "clift::CIdentifierAttr":$name,
                        "mlir::ArrayAttr":$arguments);

  let hasCustomAssemblyFormat = 1;

  let genVerifyDecl = 1;
}

def Clift_FieldAttr
  : AttrDef<Clift_Dialect,
            "Field",
            [SubElementAttrInterface]> {

  let summary = "An attribute representing a field of a struct or a union";
  let description = [{
    An attribute representing a field of a struct or a union
  }];

  let parameters = (ins StringRefParameter<>:$handle,
                        "clift::MutableStringAttr":$mutable_name,
                        "clift::MutableStringAttr":$mutable_comment,
                        "uint64_t":$offset,
                        "mlir::Type":$type);

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef NameAttrKey = "field-name";
    static constexpr llvm::StringRef CommentAttrKey = "field-comment";

    llvm::StringRef getName() const {
      return getMutableName().getValue();
    }

    llvm::StringRef getComment() const {
      return getMutableComment().getValue();
    }
  }];

  let genVerifyDecl = 1;
}

def Clift_EnumFieldAttr
  : AttrDef<Clift_Dialect,
            "EnumField",
            [SubElementAttrInterface]> {

  let summary = "An attribute representing a field of enum";
  let description = [{
    An attribute representing a field of a struct or a union
  }];

  let parameters = (ins StringRefParameter<>:$handle,
                        "clift::MutableStringAttr":$mutable_name,
                        "clift::MutableStringAttr":$mutable_comment,
                        "uint64_t":$raw_value);

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef NameAttrKey = "enum-field-name";
    static constexpr llvm::StringRef CommentAttrKey = "enum-field-comment";

    llvm::StringRef getName() const {
      return getMutableName().getValue();
    }

    llvm::StringRef getComment() const {
      return getMutableComment().getValue();
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
                        "clift::MutableStringAttr":$mutable_name,
                        "clift::MutableStringAttr":$mutable_comment,
                        "mlir::Type":$underlying_type,
                        ArrayRefParameter<"clift::EnumFieldAttr">:$fields);

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef NameAttrKey = "enum-name";
    static constexpr llvm::StringRef CommentAttrKey = "enum-comment";

    llvm::StringRef getName() const {
      return getMutableName().getValue();
    }

    llvm::StringRef getComment() const {
      return getMutableComment().getValue();
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
                        "clift::MutableStringAttr":$mutable_name,
                        "clift::MutableStringAttr":$mutable_comment,
                        "mlir::Type":$underlying_type);

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef NameAttrKey = "typedef-name";
    static constexpr llvm::StringRef CommentAttrKey = "typedef-comment";

    llvm::StringRef getName() const {
      return getMutableName().getValue();
    }

    llvm::StringRef getComment() const {
      return getMutableComment().getValue();
    }
  }];

  let genVerifyDecl = 1;
}

#endif
