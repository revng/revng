//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng-c/mlir/Dialect/Clift/IR/CliftOpTraits.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftOps.h"

using namespace mlir;
using namespace mlir::clift;

namespace traits_impl = OpTrait::clift::impl;

LogicalResult traits_impl::verifyNoFallthroughTrait(Operation *const Op) {
  if (Block *const B = Op->getBlock()) {
    Operation *const NextOp = B->getOperations().getNextNode(*Op);
    if (NextOp != nullptr and not mlir::isa<AssignLabelOp>(NextOp))
      return Op->emitOpError() << "Operation may not be followed by a non-label"
                                  "operation";
  }
  return mlir::success();
}
