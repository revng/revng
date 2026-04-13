#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/Clift.h"
#include "revng/Support/CTarget.h"

namespace clift {

mlir::LogicalResult legalizeForC(clift::FunctionOp Function,
                                 const TargetCImplementation &Target);

} // namespace clift
