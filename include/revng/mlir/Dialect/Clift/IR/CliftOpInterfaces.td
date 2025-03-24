//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_OP_INTERFACES
#define MLIR_CLIFT_OP_INTERFACES

include "mlir/IR/OpBase.td"

class Clift_OpInterface<string name> : OpInterface<name> {
  let cppNamespace = "mlir::clift";
}

def Clift_GlobalOpInterface : Clift_OpInterface<"GlobalOpInterface"> {
  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the type of the global entity.",
      /*retTy=*/"mlir::clift::ValueType",
      /*methodName=*/"getType",
      /*args=*/(ins),
      /*methodBody*/[{}],
      /*defaultImplementation=*/ [{}]
    >,
  ];
}

def Clift_StatementOpInterface : Clift_OpInterface<"StatementOpInterface"> {
}

def Clift_ExpressionOpInterface : Clift_OpInterface<"ExpressionOpInterface"> {
  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns true if the operation represents an lvalue expression.",
      /*retTy=*/"bool",
      /*methodName=*/"isLvalueExpression",
      /*args=*/(ins),
      /*methodBody*/[{}],
      /*defaultImplementation=*/ [{
        return false;
      }]
    >,
  ];
}

#endif
