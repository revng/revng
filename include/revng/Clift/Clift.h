#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <bit>

#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "revng/Clift/CliftAttributes.h"
#include "revng/Clift/CliftDialect.h"
#include "revng/Clift/CliftEnums.h"
#include "revng/Clift/CliftInterfaces.h"
#include "revng/Clift/CliftOpInterfaces.h"
#include "revng/Clift/CliftOpTraits.h"
#include "revng/Clift/CliftTypes.h"

namespace mlir::clift::impl {

inline constexpr unsigned BreakLabelFlag = 1 << 0;
inline constexpr unsigned ContinueLabelFlag = 1 << 1;

bool verifyStatementRegion(Region &R);
bool verifyExpressionRegion(Region &R, bool Required);

bool verifyPrimitiveTypeOf(ValueType Type, PrimitiveKind Kind);

unsigned getPointerArithmeticPointerOperandIndex(mlir::Operation *Op);
unsigned getPointerArithmeticOffsetOperandIndex(mlir::Operation *Op);

mlir::LogicalResult verifyUnaryIntegerMutationOp(Operation *Op);

} // namespace mlir::clift::impl

// This include should stay here for correct build procedure
#define GET_OP_CLASSES
#include "revng/Clift/Clift.h.inc"

namespace mlir::clift {

/// Returns true if the module has a Clift module attribute.
bool hasModuleAttr(mlir::ModuleOp Module);

/// Sets the Clift module attribute on the specified module.
void setModuleAttr(mlir::ModuleOp Module);

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
