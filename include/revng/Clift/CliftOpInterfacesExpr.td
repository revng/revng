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
  ];

  let extraSharedClassDeclaration = [{
    [[nodiscard]] mlir::Type getValueType() {
      return this->getOperation()->getOperand(0).getType();
    }
  }];
}

def Clift_AccessOpInterface : Clift_OpInterface<"AccessOpInterface",
                                                [Clift_ExpressionOpInterface]> {

  let description = [{
    Common interface for cast expressions.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the unique operand of this access expression.",
      "mlir::Value", "getValue", (ins)
    >,
    InterfaceMethod<
      /*desc=*/"Returns the type of the accessed class.",
      "clift::ClassType", "getClassType", (ins)
    >,
    InterfaceMethod<
      /*desc=*/"Returns the index of the accessed member.",
      "uint64_t", "getMemberIndex", (ins)
    >,
  ];

  let extraSharedClassDeclaration = [{
    // Returns the attribute describing the accessed member.
    [[nodiscard]] clift::FieldAttr getFieldAttr() {
      return this->getClassType().getFields()[this->getMemberIndex()];
    }
  }];
}

#endif
