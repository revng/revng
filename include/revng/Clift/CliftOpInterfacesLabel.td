//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_OP_INTERFACES_LABEL
#define MLIR_CLIFT_OP_INTERFACES_LABEL

include "revng/Clift/CliftOpInterfacesBasic.td"

def Clift_LabelAssignmentOpInterface
  : Clift_OpInterface<"LabelAssignmentOpInterface",
                      [Clift_StatementOpInterface]> {

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the number of jump labels assigned by this operation.",
      "unsigned", "getLabelAssignmentCount", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/ [{}]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the specified jump label assigned by this operation.",
      "mlir::Value", "getLabelAssignment", (ins "unsigned":$Index),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/ [{}]
    >,
  ];
}

#endif
