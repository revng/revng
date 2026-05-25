//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_OP_INTERFACES_BASIC
#define MLIR_CLIFT_OP_INTERFACES_BASIC

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

class Clift_OpInterface<string name, list<Interface> baseInterfaces = []>
  : OpInterface<name, baseInterfaces> {
  let cppNamespace = "clift";
}

def Clift_GlobalOpInterface : Clift_OpInterface<"GlobalOpInterface", [Symbol]> {
  let description = [{
    Common interface for all global (module-level) operations.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the type of the global entity.",
      "mlir::Type", "getType", (ins)
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
  let description = [{
    Common interface for all statement operations.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns true if the operation is indirectly non-fallthrough."
               "This is as opposed to being directly marked with the "
               "NoFallthrough trait. For example, an if-else operation whose "
               "both branches end in non-fallthrough operations is considered "
               "indirectly non-fallthrough.",
      "bool", "isIndirectlyNoFallthrough", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/ [{
        return ConcreteOp::template hasTrait<clift::NoFallthrough>();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns a local variable operation referred to by the "
               "specified block argument. The specified block argument must be "
               "defined by a region of this statement operation.",
      "mlir::Value", "getBlockArgumentVariable", (ins "mlir::BlockArgument":$Argument),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/ [{
        return nullptr;
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns true if the specified region represents a discarded "
               "expression.",
      "bool", "isDiscardedExpression", (ins "mlir::Region &":$region),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/ [{
        return false;
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns true if the specified region represents a "
               "boolean-tested expression.",
      "bool", "isBooleanTestedExpression", (ins "mlir::Region &":$region),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/ [{
        return false;
      }]
    >,
  ];
}

def Clift_ExpressionOpInterface : Clift_OpInterface<"ExpressionOpInterface"> {
  let description = [{
    Common interface for all expression operations (including yield).
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns true if the operation represents an lvalue expression.",
      "bool", "isLvalueExpression", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/ [{
        return false;
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns true if the specified operand represents a discarded "
               "subexpression.",
      "bool", "isDiscardedOperand", (ins "mlir::OpOperand &":$operand),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/ [{
        return false;
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns true if the specified operand represents a "
               "boolean-tested subexpression.",
      "bool", "isBooleanTestedOperand", (ins "mlir::OpOperand &":$operand),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/ [{
        return false;
      }]
    >,
  ];
}

def Clift_StatementRegionOpInterface
  : Clift_OpInterface<"StatementRegionOpInterface"> {

  let description = [{
    This interface describes operations containing nested statement regions and
    provides access to said regions. All present statement regions are always
    exposed, even if empty.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the number of statement regions of this operation.",
      "unsigned", "getStatementRegionCount", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/ [{}]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the specified statement region of this operation.",
      "mlir::Region &", "getStatementRegion", (ins "unsigned":$index),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/ [{}]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the statement regions of this operation.",
      "clift::StatementRegionRange", "getStatementRegions", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/ [{
        return StatementRegionRange($_op, 0, $_op.getStatementRegionCount());
      }]
    >,
  ];
}

def Clift_ExpressionRegionOpInterface
  : Clift_OpInterface<"ExpressionRegionOpInterface"> {

  let description = [{
    This interface describes operations containing nested expression regions and
    provides access to said regions. All present expression regions are always
    exposed, even if empty.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the number of expression regions of this operation.",
      "unsigned", "getExpressionRegionCount", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/ [{}]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the specified expression region of this operation.",
      "mlir::Region &", "getExpressionRegion", (ins "unsigned":$index),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/ [{}]
    >,
    InterfaceMethod<
      /*desc=*/"Returns the expression regions of this operation.",
      "clift::ExpressionRegionRange", "getExpressionRegions", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/ [{
        return ExpressionRegionRange($_op, 0, $_op.getExpressionRegionCount());
      }]
    >,
  ];
}

#endif
