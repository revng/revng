//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/CliftOpHelpers.h"

using namespace clift;

template<auto TestStatement>
static bool testExpressionUsage(YieldOp Yield) {
  mlir::Region *R = Yield->getParentRegion();
  revng_assert(R != nullptr);

  auto Statement = mlir::cast<StatementOpInterface>(R->getParentOp());
  return (Statement.*TestStatement)(*R);
}

template<auto TestStatement, auto TestOperand>
static bool testValueUsage(mlir::Value Value) {
  for (mlir::OpOperand &Use : Value.getUses()) {
    if (auto Yield = mlir::dyn_cast<YieldOp>(Use.getOwner()))
      return testExpressionUsage<TestStatement>(Yield);

    auto Expression = mlir::cast<ExpressionOpInterface>(Use.getOwner());
    if (not(Expression.*TestOperand)(Use))
      return false;
  }

  return true;
}

bool clift::isDiscarded(mlir::Value Value) {
  return testValueUsage<&StatementOpInterface::isDiscardedExpression,
                        &ExpressionOpInterface::isDiscardedOperand>(Value);
}

bool clift::isBooleanTested(mlir::Value Value) {
  return testValueUsage<&StatementOpInterface::isBooleanTestedExpression,
                        &ExpressionOpInterface::isBooleanTestedOperand>(Value);
}
