#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/Pass/Pass.h"

#include "revng/Clift/Clift.h"

namespace mlir::clift {

template<typename OpT>
using PassPtr = std::unique_ptr<mlir::OperationPass<OpT>>;

#define GEN_PASS_DECL
#include "revng/CliftTransforms/Passes.h.inc"

PassPtr<mlir::ModuleOp> createVerifyCPass();
PassPtr<mlir::ModuleOp> createEmitCPass();

#define GEN_PASS_REGISTRATION
#include "revng/CliftTransforms/Passes.h.inc"

} // namespace mlir::clift
