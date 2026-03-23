//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_OP_INTERFACES_EXPR
#define MLIR_CLIFT_OP_INTERFACES_EXPR

include "revng/Clift/CliftOpInterfacesBasic.td"

def Clift_CastOpInterface : Clift_OpInterface<"CastOpInterface",
                                              [Clift_ExpressionOpInterface]> {

  let description = [{
    Common interface for cast expressions.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the unique operand of this cast expression.",
      "mlir::Value", "getValue", (ins)
    >,
    InterfaceMethod<
      /*desc=*/"Returns the result of this cast expression.",
      "mlir::Value", "getResult", (ins)
    >,
  ];
}

#endif
