#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"
#include "revng/mlir/Dialect/Clift/Utils/CTarget.h"

namespace mlir::clift {

mlir::LogicalResult legalizeForC(clift::FunctionOp Function,
                                 const TargetCImplementation &Target);

} // namespace mlir::clift
