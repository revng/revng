//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_OP_INTERFACES_BASIC
#define MLIR_CLIFT_OP_INTERFACES_BASIC

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
  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns true if the operation is indirectly fallthrough. This "
               "is as opposed to being directly not marked with the "
               "NoFallthrough trait. For example, an if-else operation whose "
               "both branches end in non-fallthrough operations is considered "
               "indirectly non-fallthrough.",
      "bool", "hasIndirectFallthrough", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/ [{
        return not ConcreteOp::template hasTrait<mlir::OpTrait::clift::NoFallthrough>();
      }]
    >,
  ];
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
