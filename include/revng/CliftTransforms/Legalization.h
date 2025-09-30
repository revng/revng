#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/Clift.h"
#include "revng/Support/CTarget.h"

namespace mlir::clift {

mlir::LogicalResult legalizeForC(clift::FunctionOp Function,
                                 const TargetCImplementation &Target);

} // namespace mlir::clift
