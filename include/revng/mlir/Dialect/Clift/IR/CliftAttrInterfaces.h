#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/Attributes.h"

namespace mlir::clift {

class ValueType;
class ClassType;

} // namespace mlir::clift

#include "revng/mlir/Dialect/Clift/IR/CliftMutableStringAttr.h"

// Prevent reordering:
#include "revng/mlir/Dialect/Clift/IR/CliftAttrInterfaces.h.inc"
