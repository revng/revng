//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpTraits.h"

namespace traits_impl = clift::impl;

using namespace clift;
using namespace clift::impl;

static unsigned getLoopLabelMask(mlir::Operation *Op) {
  return Op->getAttrOfType<mlir::IntegerAttr>(LoopLabelMaskAttrName)
    .getValue()
    .getZExtValue();
}

static void setLoopLabelMask(mlir::Operation *Op, unsigned Mask) {
  mlir::Builder Builder(Op->getContext());
  Op->setAttr(LoopLabelMaskAttrName,
              Builder.getIntegerAttr(Builder.getIntegerType(2,
                                                            /*isSigned=*/false),
                                     Mask));
}

mlir::Value traits_impl::getLoopLabel(mlir::Operation *Op, unsigned Index) {
  unsigned Mask = getLoopLabelMask(Op);
  unsigned Flag = 1 << Index;

  if ((Mask & Flag) == 0)
    return nullptr;

  // The operand index is given by the value of the lower flag (if any):
  unsigned OperandIndex = Mask & Flag >> 1;

  return Op->getOperand(OperandIndex);
}

void traits_impl::setLoopLabel(mlir::Operation *Op,
                               unsigned Index,
                               mlir::Value Label) {

  unsigned Mask = getLoopLabelMask(Op);
  unsigned Flag = 1 << Index;

  // The operand index is given by the value of the lower flag (if any):
  unsigned OperandIndex = Mask & Flag >> 1;
  bool HasLabelOperand = Mask & Flag;

  if (HasLabelOperand != static_cast<bool>(Label))
    setLoopLabelMask(Op, Mask ^ Flag);

  if (Label) {
    if (HasLabelOperand)
      Op->setOperand(OperandIndex, Label);
    else
      Op->insertOperands(OperandIndex, { Label });
  } else if (HasLabelOperand) {
    Op->eraseOperand(OperandIndex);
  }
}

mlir::LogicalResult
traits_impl::verifyNoFallthroughTrait(mlir::Operation *const Op) {
  if (mlir::Block *const B = Op->getBlock()) {
    mlir::Operation *const NextOp = B->getOperations().getNextNode(*Op);
    if (NextOp != nullptr and not mlir::isa<AssignLabelOp>(NextOp))
      return Op->emitOpError() << "Operation may not be followed by a "
                                  "non-label operation";
  }
  return mlir::success();
}

mlir::LogicalResult
traits_impl::verifyAssignsLoopLabelsTrait(mlir::Operation *const Op) {
  constexpr auto AttrName = impl::LoopLabelMaskAttrName;
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
