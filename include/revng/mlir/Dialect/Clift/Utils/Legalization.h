#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/CTarget.h"
#include "revng/mlir/Dialect/Clift/IR/Clift.h"

namespace mlir::clift {

mlir::LogicalResult legalizeForC(clift::FunctionOp Function,
                                 const TargetCImplementation &Target);

} // namespace mlir::clift
