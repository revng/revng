#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/BuiltinOps.h"

#include "revng/CliftEmitC/Configuration.h"
#include "revng/PTML/CTokenEmitter.h"

void emitTypeAndGlobalHeader(ptml::CTokenEmitter &Tokens,
                             mlir::ModuleOp Module,
                             TypeEmitterConfiguration Configuration);

void emitHelperHeader(ptml::CTokenEmitter &Tokens,
                      llvm::ArrayRef<mlir::ModuleOp> Modules);
