#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/PatternMatch.h"

#include "revng/Clift/CliftOpHelpers.h"
#include "revng/Support/Debug.h"

namespace clift::impl {

inline mlir::Value forceBooleanResult(mlir::PatternRewriter &Rewriter,
                                      mlir::Value OldValue,
                                      mlir::Value NewValue) {
  if (not clift::isBooleanExpression(NewValue)
      and not clift::isBooleanTested(OldValue)) {
    mlir::Operation *Op = OldValue.getDefiningOp();
    revng_assert(Op);

    mlir::OpBuilder::InsertionGuard Guard(Rewriter);
    Rewriter.setInsertionPoint(Op);

    mlir::Type Type = OldValue.getType();
    mlir::Location Loc = Op->getLoc();

    mlir::Value V1 = Rewriter.create<ImmediateOp>(Loc, Type, 1);
    mlir::Value V0 = Rewriter.create<ImmediateOp>(Loc, Type, 0);
    NewValue = Rewriter.create<TernaryOp>(Loc, Type, NewValue, V1, V0);
  }
  return NewValue;
}

} // namespace clift::impl
