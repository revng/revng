//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_OP_INTERFACES_JUMP
#define MLIR_CLIFT_OP_INTERFACES_JUMP

include "revng/Clift/CliftOpInterfacesLabel.td"

def Clift_JumpStatementOpInterface
  : Clift_OpInterface<"JumpStatementOpInterface",
                      [Clift_StatementOpInterface]> {

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the target label of this jump operation.",
      "mlir::Value", "getLabel", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/ [{}]
    >,
    InterfaceMethod<
      /*desc=*/"",
      "mlir::clift::LabelAssignmentOpInterface", "getLabelAssignmentOp", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/ [{
        return mlir::clift::impl::getAssignedLabel($_op.getLabel());
      }]
    >,
  ];
}

#endif
