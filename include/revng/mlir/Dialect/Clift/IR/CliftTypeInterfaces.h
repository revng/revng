#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

namespace mlir::clift {

class FieldAttr;

} // namespace mlir::clift

#include "mlir/IR/Types.h"

#include "revng/mlir/Dialect/Clift/IR/CliftAttributes.h"

// Prevent reordering:
#include "revng/mlir/Dialect/Clift/IR/CliftTypeInterfacesBasic.h.inc"
// Prevent reordering:
#include "revng/mlir/Dialect/Clift/IR/CliftTypeInterfacesDefined.h.inc"
// Prevent reordering:
#include "revng/mlir/Dialect/Clift/IR/CliftTypeInterfacesClass.h.inc"
