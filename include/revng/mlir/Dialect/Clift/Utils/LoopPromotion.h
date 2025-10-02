#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/BuiltinOps.h"

#include "revng/Model/Binary.h"
#include "revng/mlir/Dialect/Clift/IR/CliftOpHelpers.h"

namespace mlir::clift {

void inspectLoops(FunctionOp Function);

} // namespace mlir::clift
