//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_OP_INTERFACES
#define MLIR_CLIFT_OP_INTERFACES

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

class Clift_OpInterface<string name, list<Interface> baseInterfaces = []>
  : OpInterface<name, baseInterfaces> {
  let cppNamespace = "mlir::clift";
}

def Clift_GlobalOpInterface : Clift_OpInterface<"GlobalOpInterface", [Symbol]> {
  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the type of the global entity.",
      "mlir::clift::ValueType", "getType", (ins)
    >,
    InterfaceMethod<
      /*desc=*/"Returns the user-defined handle of the entity.",
      "llvm::StringRef", "getHandle", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getHandle();
      }]
    >,
  ];
}

def Clift_StatementOpInterface : Clift_OpInterface<"StatementOpInterface"> {
}

def Clift_ExpressionOpInterface : Clift_OpInterface<"ExpressionOpInterface"> {
  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns true if the operation represents an lvalue expression.",
      "bool", "isLvalueExpression", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/ [{
        return false;
      }]
    >,
  ];
}

#endif
