//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpTraits.h"

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

LogicalResult traits_impl::verifyAssignsLoopLabelsTrait(Operation *const Op) {
  auto LabelMaskAttr = Op->getAttrOfType<mlir::IntegerAttr>("label_mask");
  revng_assert(LabelMaskAttr);

  unsigned LabelMask = LabelMaskAttr.getValue().getZExtValue();
  revng_assert(LabelMask <= 0b11);

  if (Op->getNumOperands() != static_cast<unsigned>(std::popcount(LabelMask))) {
    return Op->emitOpError() << "The number of operation operands must equal "
                                "the number of set label_mask flags.";
  }

  return mlir::success();
}
