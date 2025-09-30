#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

#include "revng/Clift/CliftInterfaces.h"
#include "revng/Support/Debug.h"

// Preserve ordering:
#include "revng/Clift/CliftDialect.h.inc"

void dumpMlirOp(mlir::Operation *Module, const char *Path) debug_function;
void dumpMlirModule(mlir::ModuleOp Module, const char *Path) debug_function;
