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

LabelAssignmentOpInterface getLabelAssignment(mlir::Value Label);

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
