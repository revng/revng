//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_TYPE_INTERFACES_INTEGRAL
#define MLIR_CLIFT_TYPE_INTERFACES_INTEGRAL

include "revng/Clift/CliftTypeInterfacesValue.td"

def Clift_IntegralType
  : Clift_TypeInterface<"IntegralType", [Clift_ValueType]> {

  let methods = [
    InterfaceMethod<
      "Returns true if the integer type is signed.",
      "bool", "isSigned", (ins)
    >,
    InterfaceMethod<
      "Returns true if the integer type is unsigned.",
      "bool", "isUnsigned", (ins)
    >,
  ];
}

#endif

