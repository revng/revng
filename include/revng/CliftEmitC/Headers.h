#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/BuiltinOps.h"

#include "revng/CliftEmitC/Configuration.h"
#include "revng/PTML/CTokenEmitter.h"

namespace mlir::clift {

void emitTypeAndGlobalHeader(ptml::CTokenEmitter &Tokens,
                             const TargetCImplementation &Target,
                             mlir::ModuleOp Module,
                             TypeEmitterConfiguration Configuration);

void emitHelperHeader(ptml::CTokenEmitter &Tokens,
                      const TargetCImplementation &Target,
                      llvm::MutableArrayRef<mlir::ModuleOp> Modules);

} // namespace mlir::clift
