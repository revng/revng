//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_TYPE_INTERFACES_DEFINED
#define MLIR_CLIFT_TYPE_INTERFACES_DEFINED

include "revng/Clift/CliftTypeInterfacesBasic.td"

def Clift_DefinedType
  : Clift_TypeInterface<"DefinedType",
                        [Clift_AliasableType, Clift_ValueType]> {

  let description = [{
    A value type is a type with a ID and Name
  }];

  let methods = [
    InterfaceMethod<
      "Returns the unique handle of the DefinedType.",
      "llvm::StringRef", "getHandle", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_type.getHandle();
      }]
    >,
    InterfaceMethod<
      "Returns the mutable name attribute of the DefinedType.",
      "mlir::clift::MutableStringAttr", "getMutableName", (ins),
      /*methodBody=*/[{}],
      /*defaultImplemetation=*/[{
        return $_type.getDefinition().getMutableName();
      }]
    >,
    InterfaceMethod<
      "Returns the name of the DefinedType.",
      "llvm::StringRef", "getName", (ins),
      /*methodBody=*/[{}],
      /*defaultImplemetation=*/[{
        return $_type.getMutableName().getValue();
      }]
    >,
  ];
}

#endif

