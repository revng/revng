#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

#include "revng/Clift/CliftOpTraits.h"
#include "revng/Clift/CliftTypes.h"

namespace mlir::clift {

class LabelAssignmentOpInterface;

namespace impl {

/// Returns the break or continue label value (if any), depending on the
/// specified index. (Index=0 for break, Index=1 for continue).
template<typename LoopOpT>
mlir::Value getLoopLabel(LoopOpT Op, unsigned Index) {
  unsigned Mask = Op.getLabelMask();
  unsigned Flag = 1 << Index;

  if ((Mask & Flag) == 0)
    return nullptr;

  // The operand index is given by the value of the lower flag (if any):
  unsigned OperandIndex = Mask & Flag >> 1;

  return Op->getOperand(OperandIndex);
}

template<typename LoopOpT>
void setLoopLabel(LoopOpT Op, unsigned Index, mlir::Value Label) {
  unsigned Mask = Op.getLabelMask();
  unsigned Flag = 1 << Index;

  // The operand index is given by the value of the lower flag (if any):
  unsigned OperandIndex = Mask & Flag >> 1;
  bool HasLabelOperand = Mask & Flag;

  if (HasLabelOperand != static_cast<bool>(Label))
    Op.setLabelMask(Mask);

  if (Label) {
    if (HasLabelOperand)
      Op->setOperand(OperandIndex, Label);
    else
      Op->insertOperands(OperandIndex, { Label });
  } else if (HasLabelOperand) {
    Op->eraseOperand(OperandIndex);
  }
}

LabelAssignmentOpInterface getAssignedLabel(mlir::Value Label);

} // namespace impl
} // namespace mlir::clift

// Prevent reordering:
#include "revng/Clift/CliftOpInterfacesBasic.h.inc"
// Prevent reordering:
#include "revng/Clift/CliftOpInterfacesLabel.h.inc"
// Prevent reordering:
#include "revng/Clift/CliftOpInterfacesJump.h.inc"
// Prevent reordering:
#include "revng/Clift/CliftOpInterfacesControlFlow.h.inc"

namespace mlir::clift {

bool isLvalueExpression(mlir::Value Value);

} // namespace mlir::clift
