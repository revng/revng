#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/Clift.h"

namespace clift {

mlir::LogicalResult legalizeForC(clift::FunctionOp Function);

} // namespace clift
