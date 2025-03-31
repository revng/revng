//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_TYPE_INTERFACES_DEFINED
#define MLIR_CLIFT_TYPE_INTERFACES_DEFINED

include "revng/mlir/Dialect/Clift/IR/CliftTypeInterfacesBasic.td"

def Clift_DefinedType
  : Clift_TypeInterface<"DefinedType",
                        [Clift_AliasableType, Clift_ValueType]> {

  let description = [{
    A value type is a type with a ID and Name
  }];

  let methods = [
    InterfaceMethod<
      "The unique handle of the type",
      "llvm::StringRef", "getHandle", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_type.getHandle();
      }]
    >,
    InterfaceMethod<
      "returns the name of the type.",
      "llvm::StringRef", "getName", (ins),
      /*methodBody=*/[{}],
      /*defaultImplemetation=*/[{
        return $_type.getName();
      }]
    >,
  ];
}

#endif

