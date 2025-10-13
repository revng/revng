#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Module.h"

#include "revng/Clift/Clift.h"
#include "revng/Model/Binary.h"

namespace mlir::clift {

class Clifter {
public:
  static std::unique_ptr<Clifter> make(mlir::ModuleOp Module,
                                       const model::Binary &Model);

  virtual ~Clifter() = default;

  virtual clift::FunctionOp import(const llvm::Function *Function) = 0;

protected:
  Clifter() = default;
  Clifter(const Clifter &) = default;
  Clifter &operator=(const Clifter &) = default;
};

} // namespace mlir::clift
