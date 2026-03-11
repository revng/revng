//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_TYPE_INTERFACES_PRIMITIVE
#define MLIR_CLIFT_TYPE_INTERFACES_PRIMITIVE

include "revng/Clift/CliftTypeInterfacesObject.td"

def Clift_PrimitiveType
  : Clift_TypeInterface<"PrimitiveType",
                        [Clift_AliasableType, Clift_ValueType]> {
}

#endif

