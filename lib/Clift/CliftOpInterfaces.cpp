//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpInterfaces.h"

// Prevent reordering:
#include "revng/Clift/CliftOpInterfacesBasic.cpp.inc"
// Prevent reordering:
#include "revng/Clift/CliftOpInterfacesLabel.cpp.inc"
// Prevent reordering:
#include "revng/Clift/CliftOpInterfacesJump.cpp.inc"
// Prevent reordering:
#include "revng/Clift/CliftOpInterfacesControlFlow.cpp.inc"
// Prevent reordering:
#include "revng/Clift/CliftOpInterfacesExpr.cpp.inc"

using namespace clift;

LabelAssignmentOpInterface
clift::impl::getLabelAssignmentOp(mlir::Value Label) {
  // A promoted (operand-less) break_to/continue_to has no target label.
  if (not Label)
    return {};
  return Label.getDefiningOp<MakeLabelOp>().getAssignment();
}

bool clift::isLvalueExpression(mlir::Value Value) {
  if (auto Argument = mlir::dyn_cast<mlir::BlockArgument>(Value)) {
    mlir::Block *B = Argument.getOwner();
    if (B == nullptr)
      return false;

    mlir::Operation *Op = B->getParentOp();
    if (Op == nullptr)
      return false;

    // Function block arguments represent parameters. All statement block
    // arguments refer to local variables, and all local variables are l-values.
    return mlir::isa<FunctionOp, StatementOpInterface>(Op);
  }

  mlir::Operation *Op = Value.getDefiningOp();
  if (auto E = mlir::dyn_cast<ExpressionOpInterface>(Op))
    return E.isLvalueExpression();

  return mlir::isa<LocalVariableOp>(Op);
}
