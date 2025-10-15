//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_INTERFACE
#define MLIR_CLIFT_INTERFACE

include "mlir/IR/OpBase.td"

class Clift_AttrInterface<string name, list<Interface> baseInterfaces = []>
  : AttrInterface<name, baseInterfaces> {
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

def Clift_TypeDefinitionAttr : Clift_AttrInterface<"TypeDefinitionAttr"> {
  let methods = [
    InterfaceMethod<
      "Returns the unique handle of the DefinedType.",
      "llvm::StringRef", "getHandle", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_attr.getHandle();
      }]
    >,
    InterfaceMethod<
      "Returns the mutable name attribute of the DefinedType.",
      "mlir::clift::MutableStringAttr", "getMutableName", (ins),
      /*methodBody=*/[{}],
      /*defaultImplemetation=*/[{}]
    >,
    InterfaceMethod<
      "Returns the name of the DefinedType.",
      "llvm::StringRef", "getName", (ins),
      /*methodBody=*/[{}],
      /*defaultImplemetation=*/[{
        return $_attr.getMutableName().getValue();
      }]
    >,
  ];
}

#endif

