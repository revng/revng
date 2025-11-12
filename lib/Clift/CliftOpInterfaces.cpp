/// \file CliftOpInterfaces.cpp
/// Tests for the Clift Dialect

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
//
#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpInterfaces.h"

namespace mlir {

// Prevent reordering:
#include "revng/Clift/CliftOpInterfacesBasic.cpp.inc"
// Prevent reordering:
#include "revng/Clift/CliftOpInterfacesLabel.cpp.inc"
// Prevent reordering:
#include "revng/Clift/CliftOpInterfacesJump.cpp.inc"
// Prevent reordering:
#include "revng/Clift/CliftOpInterfacesControlFlow.cpp.inc"

} // namespace mlir

namespace clift = mlir::clift;
using namespace clift;

LabelAssignmentOpInterface
clift::impl::getLabelAssignmentOp(mlir::Value Label) {
  return Label.getDefiningOp<MakeLabelOp>().getAssignment();
}

bool clift::isLvalueExpression(mlir::Value Value) {
  if (auto Argument = mlir::dyn_cast<mlir::BlockArgument>(Value)) {
    Block *B = Argument.getOwner();
    if (B == nullptr)
      return false;

    mlir::Operation *Op = B->getParentOp();
    if (Op == nullptr)
      return false;

    return mlir::isa<FunctionOp, LoopOpInterface>(Op);
  } else {
    mlir::Operation *Op = Value.getDefiningOp();

    if (auto ExprOp = mlir::dyn_cast<ExpressionOpInterface>(Value
                                                              .getDefiningOp()))
      return ExprOp.isLvalueExpression();

    return mlir::isa<LocalVariableOp, GlobalVariableOp>(Op);
  }
}
