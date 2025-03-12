#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "revng/mlir/Dialect/Clift/IR/Clift.h"
#include "revng/mlir/Dialect/Clift/IR/CliftAttributes.h"
#include "revng/mlir/Dialect/Clift/IR/CliftEnums.h"
#include "revng/mlir/Dialect/Clift/IR/CliftInterfaces.h"
#include "revng/mlir/Dialect/Clift/IR/CliftOpTraits.h"
#include "revng/mlir/Dialect/Clift/IR/CliftTypes.h"

namespace mlir::clift::impl {

bool verifyStatementRegion(Region &R);
bool verifyExpressionRegion(Region &R, bool Required);

bool verifyPrimitiveTypeOf(ValueType Type, PrimitiveKind Kind);

mlir::LogicalResult verifyUnaryIntegerMutationOp(Operation *Op);

} // namespace mlir::clift::impl

// This include should stay here for correct build procedure
#define GET_OP_CLASSES
#include "revng/mlir/Dialect/Clift/IR/CliftOps.h.inc"

namespace mlir::clift {

/// Returns the terminating YieldOp of the expression represented by the region,
/// or a operation if the region is not a valid expression region.
YieldOp getExpressionYieldOp(Region &R);

/// Returns the value of the expression represented by the region region, or a
/// null value if the region is not a valid expression region.
mlir::Value getExpressionValue(Region &R);

/// Returns the type of the expression represented by the region, or a null type
/// if region is not a valid expression region.
clift::ValueType getExpressionType(Region &R);

} // namespace mlir::clift
