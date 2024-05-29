#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLFunctionalExtras.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionInterfaces.h"

#include "revng/Model/Binary.h"

namespace mlir::clift {

void importReachableModelTypes(mlir::ModuleOp Module,
                               const model::Binary &Model);

} // namespace mlir::clift
