//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_TYPE_INTERFACES_OBJECT
#define MLIR_CLIFT_TYPE_INTERFACES_OBJECT

include "revng/Clift/CliftTypeInterfacesBasic.td"

def Clift_ObjectType : Clift_TypeInterface<"ObjectType", [Clift_ValueType]> {
  let description = [{
    The object types is the set of types usable as types of variables and
    fields.
  }];

  let methods = [
    InterfaceMethod<
      "Returns the size of the object type.",
      "uint64_t", "getObjectSize", (ins)
    >,
  ];
}

#endif

