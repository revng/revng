#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng-c/mlir/Dialect/Clift/Transforms/ModelOption.h"

namespace mlir::clift {

#define GEN_PASS_DECL
#include "revng-c/mlir/Dialect/Clift/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "revng-c/mlir/Dialect/Clift/Transforms/Passes.h.inc"

} // namespace mlir::clift
