/// \file CliftOpInterfaces.cpp
/// Tests for the Clift Dialect

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
//
#include "revng/mlir/Dialect/Clift/IR/CliftOpHelpers.h"
#include "revng/mlir/Dialect/Clift/IR/CliftOpInterfaces.h"
#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"

namespace mlir {
#include "revng/mlir/Dialect/Clift/IR/CliftOpInterfacesBasic.cpp.inc"
//
#include "revng/mlir/Dialect/Clift/IR/CliftOpInterfacesStatement.cpp.inc"
} // namespace mlir

namespace clift = mlir::clift;

bool clift::isLvalueExpression(mlir::Value Value) {
  if (auto Argument = mlir::dyn_cast<mlir::BlockArgument>(Value)) {
    Block *B = Argument.getOwner();
    if (B == nullptr)
      return false;

    mlir::Operation *Op = B->getParentOp();
    if (Op == nullptr)
      return false;

    return mlir::isa<FunctionOp>(Op);
  } else {
    mlir::Operation *Op = Value.getDefiningOp();

    if (auto ExprOp = mlir::dyn_cast<ExpressionOpInterface>(Value
                                                              .getDefiningOp()))
      return ExprOp.isLvalueExpression();

    return mlir::isa<LocalVariableOp, GlobalVariableOp>(Op);
  }
}
