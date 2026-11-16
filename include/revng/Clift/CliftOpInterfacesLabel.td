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
      "unsigned", "getAssignedLabelCount", (ins)
    >,
    InterfaceMethod<
      /*desc=*/"Returns the specified jump label assigned by this operation.",
      "mlir::Value", "getAssignedLabel", (ins "unsigned":$Index)
    >,
    InterfaceMethod<
      /*desc=*/"Returns the label assigned by this operation, where execution "
               "continues as if after falling through, or null if no such "
               "label is assigned.",
      "mlir::Value", "getTrailingLabel", (ins)
    >,
  ];
}

#endif
