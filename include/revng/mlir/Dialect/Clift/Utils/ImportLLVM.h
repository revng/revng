#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Module.h"

#include "revng/Model/Binary.h"
#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"

namespace mlir::clift {

class LLVMToCliftImporter {
public:
  static std::unique_ptr<LLVMToCliftImporter> make(mlir::ModuleOp Module,
                                                   const model::Binary &Model);

  virtual ~LLVMToCliftImporter() = default;

  virtual clift::FunctionOp import(const llvm::Function *Function) = 0;

protected:
  LLVMToCliftImporter() = default;
  LLVMToCliftImporter(const LLVMToCliftImporter &) = default;
  LLVMToCliftImporter &operator=(const LLVMToCliftImporter &) = default;
};

} // namespace mlir::clift
