//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_OP_INTERFACES_STATEMENT
#define MLIR_CLIFT_OP_INTERFACES_STATEMENT

include "revng/mlir/Dialect/Clift/IR/CliftOpInterfacesBasic.td"

def Clift_BranchOpInterface
  : Clift_OpInterface<"BranchOpInterface", [Clift_StatementOpInterface]> {

  let methods = [
    InterfaceMethod<
      /*desc=*/"",
      "mlir::Region &", "getBranchCondition", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/ [{
        return $_op.getCondition();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"",
      "llvm::MutableArrayRef<mlir::Region>", "getBranchRegions", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/ [{
        return $_op->getRegions().drop_front(1);
      }]
    >,
  ];
}

def Clift_LoopOpInterface
  : Clift_OpInterface<"LoopOpInterface", [Clift_StatementOpInterface]> {

  let methods = [
    InterfaceMethod<
      /*desc=*/"",
      "mlir::Region &", "getLoopRegion", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getBody();
      }]
    >
  ];
}

#endif

