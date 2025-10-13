//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_OP_INTERFACES_CONTROL_FLOW
#define MLIR_CLIFT_OP_INTERFACES_CONTROL_FLOW

include "revng/Clift/CliftOpInterfacesJump.td"

def Clift_BranchOpInterface : Clift_OpInterface<"BranchOpInterface",
                                                [Clift_StatementOpInterface]> {

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the region describing the branch condition.",
      "mlir::Region &", "getBranchCondition", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/ [{
        return $_op.getCondition();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the regions describing the branches of this operation.",
      "llvm::MutableArrayRef<mlir::Region>", "getBranchRegions", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/ [{
        return $_op->getRegions().drop_front(1);
      }]
    >,
  ];
}

#endif
