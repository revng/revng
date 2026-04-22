#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/BuiltinOps.h"

#include "revng/Model/Binary.h"

namespace mlir::clift {

mlir::LogicalResult verifyAgainstModel(mlir::ModuleOp Module,
                                       const model::Binary &Model);

} // namespace mlir::clift
