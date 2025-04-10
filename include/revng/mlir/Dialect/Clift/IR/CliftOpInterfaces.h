#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

#include "revng/mlir/Dialect/Clift/IR/CliftTypes.h"

// This comment prevents reordering this include before the others.
#include "revng/mlir/Dialect/Clift/IR/CliftOpInterfaces.h.inc"

namespace mlir::clift {

bool isLvalueExpression(mlir::Value Value);

} // namespace mlir::clift
