#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/CliftContainers.h"
#include "revng/PipeboxCommon/Helpers/PipeRuns/CliftFunctionMixin.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::piperuns {

class ModelVerifyClift : public CliftFunctionMixin<ModelVerifyClift> {
private:
  const model::Binary &Binary;

public:
  static constexpr llvm::StringRef Name = "model-verify-clift";
  using Arguments = TypeList<PipeRunArgument<CliftFunctionContainer,
                                             "Modules",
                                             "function MLIR module(s)">>;

  ModelVerifyClift(const class Model &Model,
                   llvm::StringRef StaticConfiguration,
                   llvm::StringRef Configuration,
                   CliftFunctionContainer &ModuleContainer) :
    CliftFunctionMixin(ModuleContainer), Binary(*Model.get().get()) {}

  void runOnCliftFunction(const model::Function &Function,
                          clift::FunctionOp MLIRFunction);
};

} // namespace revng::pypeline::piperuns
