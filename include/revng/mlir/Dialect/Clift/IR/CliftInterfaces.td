//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_INTERFACE
#define MLIR_CLIFT_INTERFACE

include "mlir/IR/OpBase.td"

class Clift_TypeInterface<string name> : TypeInterface<name> {
  let cppNamespace = "mlir::clift";
}

class Clift_AttrInterface<string name> : AttrInterface<name> {
  let cppNamespace = "mlir::clift";
}

def Clift_AliasableAttr : Clift_AttrInterface<"AliasableAttr"> {
  let description = [{
    A Aliasable type is a type with a size and a constness
  }];

  let methods = [
    InterfaceMethod<
      "Writes the alias to OS and returns true if it is non-empty.",
      "bool", "getAlias", (ins "llvm::raw_ostream &":$OS),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{}]
    >,
  ];
}

def Clift_AliasableType : Clift_TypeInterface<"AliasableType"> {
  let description = [{
    A Aliasable type is a type with a size and a constness
  }];

  let methods = [
    InterfaceMethod<
      "Writes the alias to OS and returns true if it is non-empty.",
      "bool", "getAlias", (ins "llvm::raw_ostream &":$OS),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{}]
    >,
  ];
}

def Clift_SizedType : Clift_AttrInterface<"SizedType"> {
  let description = [{
    A sized type is a type with a size
  }];

  let methods = [
    InterfaceMethod<
      "returns the size of the type.",
      "uint64_t", "getByteSize", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_attr.getSize();
      }]
    >,
  ];
}

def Clift_ValueType : Clift_TypeInterface<"ValueType"> {
  let description = [{
    A value type is a type with a size and a constness
  }];

  let methods = [
    InterfaceMethod<
      "returns the size of the type.",
      "uint64_t", "getByteSize", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_type.getSize();
      }]
    >,
    InterfaceMethod<
      "the constness of the type",
      "bool", "isConst", (ins),
      /*methodBody=*/[{}],
      /*defaultImplemetation=*/[{
        return $_type.getIsConst().getValue();
      }]
    >,
    InterfaceMethod<
      "Returns the unqualified base type of the type.",
      "mlir::clift::ValueType", "addConst", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{}]
    >,
    InterfaceMethod<
      "Returns the unqualified base type of the type.",
      "mlir::clift::ValueType", "removeConst", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{}]
    >,
  ];
}

def Clift_TypeDefinitionAttr : Clift_AttrInterface<"TypeDefinitionAttr"> {
  let description = [{
    A value type is a type with a ID and Name
  }];

  let methods = [
    InterfaceMethod<
      "The unique handle of the type",
      "llvm::StringRef", "getUniqueHandle", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_attr.getUniqueHandle();
      }]
    >,
    InterfaceMethod<
      "returns the name of the type.",
      "llvm::StringRef", "name", (ins),
      /*methodBody=*/[{}],
      /*defaultImplemetation=*/[{
        return $_attr.getName();
      }]
    >,
    InterfaceMethod<
      "returns the size of the type.",
      "uint64_t", "getByteSize", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_attr.getSize();
      }]
    >,
  ];
}

def Clift_ClassTypeAttr : Clift_AttrInterface<"ClassTypeAttr"> {
  let methods = [
    InterfaceMethod<
      "Returns the field attributes of this class type",
      "llvm::ArrayRef<mlir::clift::FieldAttr>",
      "getFields",
      (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_attr.getFields();
      }]
    >
  ];
}

#endif

