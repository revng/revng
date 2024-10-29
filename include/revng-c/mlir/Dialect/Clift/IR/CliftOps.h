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

#include "revng-c/mlir/Dialect/Clift/IR/Clift.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftAttributes.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftEnums.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftInterfaces.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftOpTraits.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftTypes.h"

namespace mlir::clift::impl {

bool verifyStatementRegion(Region &R);
bool verifyExpressionRegion(Region &R, bool Required);

bool verifyPrimitiveTypeOf(ValueType Type, PrimitiveKind Kind);

mlir::Type removeCliftConst(mlir::Type Type);

} // namespace mlir::clift::impl

// This include should stay here for correct build procedure
#define GET_OP_CLASSES
#include "revng-c/mlir/Dialect/Clift/IR/CliftOps.h.inc"
