//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Include the definition of the necessary tablegen constructs for defining our
// types
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SubElementInterfaces.td"
include "Clift.td"
include "CliftAttributes.td"
include "CliftInterfaces.td"

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

def Clift_PrimitiveType : Clift_Type<"Primitive", "primitive", [Clift_ValueType, DeclareTypeInterfaceMethods<Clift_AliasableType>]> {
  let summary = "Clift primitive type";

  let parameters = (ins "PrimitiveKind":$kind, "uint64_t":$size, DefaultValuedParameter<"BoolAttr", "BoolAttr::get($_ctxt, false)">:$is_const);

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    static mlir::clift::PrimitiveType getVoid(mlir::MLIRContext* Context, uint64_t Size, bool IsConst = false) {
      return get(Context, PrimitiveKind::VoidKind, Size, BoolAttr::get(Context, IsConst));
    }
  }];

  let description = [{
    Primitive type.
  }];

  let assemblyFormat = "`<`( `is_const` `=` $is_const^ `,`` `)? $kind $size `>` ";
}


def Clift_PointerType : Clift_Type<"Pointer", "pointer", [DeclareTypeInterfaceMethods<Clift_ValueType, ["getByteSize"]>, SubElementTypeInterface]> {
  let summary = "Clift pointer type";

  let parameters = (ins "mlir::clift::ValueType":$pointee_type, "uint64_t":$pointer_size, DefaultValuedParameter<"BoolAttr", "BoolAttr::get($_ctxt, false)">:$is_const);


  let description = [{
    Pointer type.
  }];
  let genVerifyDecl = 1;

  let assemblyFormat = "`<`( `is_const` `=` $is_const^ `,`` ` )? struct($pointee_type, $pointer_size) `>` ";
}

def Clift_DefinedType : Clift_Type<"Defined", "defined", [DeclareTypeInterfaceMethods<Clift_ValueType, ["getByteSize", "id"]>, SubElementTypeInterface, DeclareTypeInterfaceMethods<Clift_AliasableType>]> {
  let summary = "Clift nominal type";

  let parameters = (ins "mlir::clift::TypeDefinitionAttr":$element_type, DefaultValuedParameter<"BoolAttr", "BoolAttr::get($_ctxt, false)">:$is_const);

  let description = [{
     A defined type wraps a TypeDefinitionAttr attribute so that it can be attached
     to values. Defined type exists because we are mirroring the revng model
     type hierarchy which has the same concept. TypeDefinitions are attributes
     to prevent them from being attached to values without being wrapped into
     a defined type.
  }];
  let extraClassDeclaration = [{
    uint64_t id() const;
    llvm::StringRef name() const;
  }];
  let genVerifyDecl = 1;

  let assemblyFormat = "`<`( `is_const` `=` $is_const^ `,`` ` )? $element_type `>` ";
}

def Clift_ArrayType : Clift_Type<"Array", "array", [DeclareTypeInterfaceMethods<Clift_ValueType, ["getByteSize"]>, SubElementTypeInterface]> {
  let summary = "Clift Array type";

  let parameters = (ins "mlir::clift::ValueType":$element_type, "uint64_t":$elements_count);

  let description = [{
    ArrayType type.
  }];
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    BoolAttr getIsConst() const {
      return BoolAttr::get(getContext(), false);
    }
  }];

  let assemblyFormat = "`<` struct($element_type, $elements_count) `>` ";
}
