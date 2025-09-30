#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/CTarget.h"
#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"

namespace mlir::clift {

mlir::LogicalResult verifyCSemantics(mlir::ModuleOp Module,
                                     const TargetCImplementation &Target);

} // namespace mlir::clift
