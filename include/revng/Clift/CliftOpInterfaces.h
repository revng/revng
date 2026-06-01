#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

#include "revng/Clift/CliftOpTraits.h"
#include "revng/Clift/CliftTypes.h"

namespace clift {

class LabelAssignmentOpInterface;
class StatementRegionOpInterface;
class ExpressionRegionOpInterface;

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
    Op.setLabelMask(Mask ^ Flag);

  if (Label) {
    if (HasLabelOperand)
      Op->setOperand(OperandIndex, Label);
    else
      Op->insertOperands(OperandIndex, { Label });
  } else if (HasLabelOperand) {
    Op->eraseOperand(OperandIndex);
  }
}

LabelAssignmentOpInterface getLabelAssignmentOp(mlir::Value Label);

} // namespace impl

enum class LvalueToRvalueConversion : uint8_t {
  // The operand is not subject to l-value-to-r-value conversion.
  No,

  // The operand is subject to l-value-to-r-value conversion.
  Yes,

  // The operand is subject to l-value-to-r-value conversion if the result of
  // its owner is subject to l-value-to-r-value conversion.
  Transparent,
};

/// Provides a range over the statement regions of an operation by utilizing the
/// index-based interface of StatementRegionOpInterface.
struct StatementRegionRange : llvm::indexed_accessor_range<StatementRegionRange,
                                                           mlir::Operation *,
                                                           mlir::Region> {

  StatementRegionRange() :
    indexed_accessor_range(static_cast<mlir::Operation *>(nullptr), 0, 0) {}

  using indexed_accessor_range::indexed_accessor_range;

  static mlir::Region &dereference(mlir::Operation *Op, ptrdiff_t Index);
};

/// Provides a range over the expression regions of an operation by utilizing
/// the index-based interface of ExpressionRegionOpInterface.
struct ExpressionRegionRange
  : llvm::indexed_accessor_range<ExpressionRegionRange,
                                 mlir::Operation *,
                                 mlir::Region> {

  ExpressionRegionRange() :
    indexed_accessor_range(static_cast<mlir::Operation *>(nullptr), 0, 0) {}

  using indexed_accessor_range::indexed_accessor_range;

  static mlir::Region &dereference(mlir::Operation *Op, ptrdiff_t Index);
};

} // namespace clift

// Prevent reordering:
#include "revng/Clift/CliftOpInterfacesBasic.h.inc"
// Prevent reordering:
#include "revng/Clift/CliftOpInterfacesLabel.h.inc"
// Prevent reordering:
#include "revng/Clift/CliftOpInterfacesJump.h.inc"
// Prevent reordering:
#include "revng/Clift/CliftOpInterfacesControlFlow.h.inc"
// Prevent reordering:
#include "revng/Clift/CliftOpInterfacesExpr.h.inc"

namespace clift {

inline mlir::Region &StatementRegionRange::dereference(mlir::Operation *Op,
                                                       ptrdiff_t Index) {
  auto Regions = mlir::cast<StatementRegionOpInterface>(Op);
  revng_assert(Index >= 0);
  revng_assert(Index <= Regions.getStatementRegionCount());
  return Regions.getStatementRegion(static_cast<unsigned>(Index));
}

inline mlir::Region &ExpressionRegionRange::dereference(mlir::Operation *Op,
                                                        ptrdiff_t Index) {
  auto Regions = mlir::cast<ExpressionRegionOpInterface>(Op);
  revng_assert(Index >= 0);
  revng_assert(Index <= Regions.getExpressionRegionCount());
  return Regions.getExpressionRegion(static_cast<unsigned>(Index));
}

bool isLvalueExpression(mlir::Value Value);

} // namespace clift
