//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/CliftOpHelpers.h"

namespace clift = mlir::clift;
using namespace clift;

template<auto TestOperand>
static bool testValueUsage(mlir::Value Value) {
  for (mlir::OpOperand &Use : Value.getUses()) {
    auto Expression = mlir::cast<ExpressionOpInterface>(Use.getOwner());
    if (not(Expression.*TestOperand)(Use))
      return false;
  }

  return true;
}

bool clift::isDiscarded(mlir::Value Value) {
  return testValueUsage<&ExpressionOpInterface::isDiscardedOperand>(Value);
}

bool clift::isBooleanTested(mlir::Value Value) {
  return testValueUsage<&ExpressionOpInterface::isBooleanTestedOperand>(Value);
}
