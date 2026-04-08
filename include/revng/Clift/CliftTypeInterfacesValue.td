//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_TYPE_INTERFACES_VALUE
#define MLIR_CLIFT_TYPE_INTERFACES_VALUE

include "revng/Clift/CliftTypeInterfacesObject.td"

def Clift_ValueType : Clift_TypeInterface<"ValueType", [Clift_ObjectType]> {
  let description = [{
    The value types is the set of types which can be assigned, passed as
    function parameters, returned from functions (along with void), etc...
  }];
}

#endif

