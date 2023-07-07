//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "mlir/IR/RegionGraphTraits.h"

#include "revng/Support/GraphAlgorithms.h"

#include "revng-c/mlir/Dialect/Clift/IR/CliftOps.h"

#define GET_OP_CLASSES
#include "revng-c/mlir/Dialect/Clift/IR/CliftOps.cpp.inc"

void mlir::clift::CliftDialect::registerOperations() {
  addOperations</* Include the auto-generated clift operations */
#define GET_OP_LIST
#include "revng-c/mlir/Dialect/Clift/IR/CliftOps.cpp.inc"
                /* End of operations list */>();
}

//===-----------------------------------------------------------------========//
// Code for clift::AssignLabelOp.
//===----------------------------------------------------------------------===//

mlir::LogicalResult
mlir::clift::AssignLabelOp::canonicalize(mlir::clift::AssignLabelOp Op,
                                         mlir::PatternRewriter &Rewriter) {
  for (const mlir::OpOperand &Use : Op.getLabel().getUses())
    if (mlir::isa<mlir::clift::GoToOp>(Use.getOwner()))
      return mlir::success();

  Rewriter.eraseOp(Op);
  return mlir::success();
}

//===-----------------------------------------------------------------========//
// Code for clift::MakeLabelOp.
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::clift::MakeLabelOp::verify() {

  auto AssignOwnerLambda = [](mlir::OpOperand &Operand) {
    return mlir::isa<mlir::clift::AssignLabelOp>(Operand.getOwner());
  };
  const bool HasAssign = llvm::any_of(getResult().getUses(), AssignOwnerLambda);

  if (HasAssign)
    return mlir::success();

  auto GoToOwnerLambda = [](mlir::OpOperand &Operand) {
    return mlir::isa<mlir::clift::GoToOp>(Operand.getOwner());
  };
  const bool HasGoTo = llvm::any_of(getResult().getUses(), GoToOwnerLambda);

  if (not HasGoTo)
    return mlir::success();

  emitOpError(getOperationName() + " with a "
              + mlir::clift::GoToOp::getOperationName() + " use must have a "
              + mlir::clift::AssignLabelOp::getOperationName() + " use too.");
  return mlir::failure();
}

//===-----------------------------------------------------------------========//
// Code for clift::LoopOp.
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::clift::LoopOp::verifyRegions() {

  // Verify that the region inside each `clift.loop` is acyclic

  // TODO: this verifier does not cover the root region, because that is not
  //       inside a `clift.loop`. A solution to this may be the definition of a
  //       `IsCombableInterface` in order to perform the verification on the
  //       interface and not on the operation.
  mlir::Region &LoopOpRegion = getBody();
  if (isDAG(&LoopOpRegion)) {
    return success();
  } else {
    return failure();
  }
}
