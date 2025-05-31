//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_TYPE_INTERFACES_BASIC
#define MLIR_CLIFT_TYPE_INTERFACES_BASIC

include "mlir/IR/OpBase.td"

class Clift_TypeInterface<string name, list<Interface> baseInterfaces = []>
  : TypeInterface<name, baseInterfaces> {
  let cppNamespace = "mlir::clift";
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

def Clift_ValueType : Clift_TypeInterface<"ValueType"> {
  let description = [{
    A value type is a type with a size and a constness
  }];

  let methods = [
    InterfaceMethod<
      "returns the size of the type.",
      "uint64_t", "getByteSize", (ins),
      /*methodBody=*/[{}],
      /*defaultImplemetation=*/[{
        return $_type.getSize();
      }]
    >,
    InterfaceMethod<
      "the constness of the type",
      "bool", "isConst", (ins),
      /*methodBody=*/[{}],
      /*defaultImplemetation=*/[{
        return $_type.getIsConst();
      }]
    >,
    InterfaceMethod<
      "Returns the unqualified base type of the type.",
      "mlir::clift::ValueType", "addConst", (ins)
    >,
    InterfaceMethod<
      "Returns the unqualified base type of the type.",
      "mlir::clift::ValueType", "removeConst", (ins)
    >,
  ];
}

#endif

