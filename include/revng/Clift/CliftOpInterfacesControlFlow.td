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

def Clift_LoopOpInterface
  : Clift_OpInterface<"LoopOpInterface", [Clift_LabelAssignmentOpInterface]> {

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the loop break label (if any).",
      "mlir::Value", "getBreakLabel", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return mlir::clift::impl::getLoopLabel($_op, 0);
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Replaces the loop break label (if any).",
      "void", "setBreakLabel", (ins "mlir::Value":$Label),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return mlir::clift::impl::setLoopLabel($_op, 0, Label);
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the loop continue label (if any).",
      "mlir::Value", "getContinueLabel", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return mlir::clift::impl::getLoopLabel($_op, 1);
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Replaces the loop continue label (if any).",
      "void", "setContinueLabel", (ins "mlir::Value":$Label),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return mlir::clift::impl::setLoopLabel($_op, 1, Label);
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the loop body region.",
      "mlir::Region &", "getBody", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getBody();
      }]
    >,
  ];
}

#endif
