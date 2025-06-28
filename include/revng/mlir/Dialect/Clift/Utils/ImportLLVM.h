#pragma once

#include "llvm/IR/Module.h"

#include "revng/Model/Binary.h"
#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"

namespace mlir::clift {

void importLLVM(mlir::ModuleOp ModuleOp,
                const model::Binary &Model,
                const llvm::Module *Module);

mlir::OwningOpRef<mlir::ModuleOp> importLLVM(mlir::MLIRContext *Context,
                                             const model::Binary &Model,
                                             const llvm::Module *Module);

} // namespace mlir::clift
