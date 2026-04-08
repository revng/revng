//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpTraits.h"

namespace traits_impl = mlir::OpTrait::clift::impl;

namespace clift = mlir::clift;
using namespace clift;

mlir::LogicalResult
traits_impl::verifyNoFallthroughTrait(mlir::Operation *const Op) {
  if (mlir::Block *const B = Op->getBlock()) {
    mlir::Operation *const NextOp = B->getOperations().getNextNode(*Op);
    if (NextOp != nullptr and not mlir::isa<AssignLabelOp>(NextOp))
      return Op->emitOpError() << "Operation may not be followed by a non-label"
                                  "operation";
  }
  return mlir::success();
}

mlir::LogicalResult
traits_impl::verifyAssignsLoopLabelsTrait(mlir::Operation *const Op) {
  constexpr auto AttrName = mlir::clift::impl::LoopLabelMaskAttrName;
  auto LabelMaskAttr = Op->getAttrOfType<mlir::IntegerAttr>(AttrName);
  revng_assert(LabelMaskAttr);

  unsigned LabelMask = LabelMaskAttr.getValue().getZExtValue();
  revng_assert(LabelMask <= 0b11);

  if (Op->getNumOperands() != static_cast<unsigned>(std::popcount(LabelMask))) {
    return Op->emitOpError() << "The number of operation operands must equal "
                                "the number of set label_mask flags.";
  }

  return mlir::success();
}
