#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

#include "revng/Support/Debug.h"
#include "revng/mlir/Dialect/Clift/IR/CliftInterfaces.h"

// Preserve ordering:
#include "revng/mlir/Dialect/Clift/IR/CliftOpsDialect.h.inc"

void dumpMlirOp(mlir::Operation *Module, const char *Path) debug_function;
void dumpMlirModule(mlir::ModuleOp Module, const char *Path) debug_function;
