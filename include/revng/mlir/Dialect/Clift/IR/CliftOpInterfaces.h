#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

#include "revng/mlir/Dialect/Clift/IR/CliftOpTraits.h"
#include "revng/mlir/Dialect/Clift/IR/CliftTypes.h"

//
#include "revng/mlir/Dialect/Clift/IR/CliftOpInterfacesBasic.h.inc"
//
#include "revng/mlir/Dialect/Clift/IR/CliftOpInterfacesStatement.h.inc"

namespace mlir::clift {

bool isLvalueExpression(mlir::Value Value);

} // namespace mlir::clift
