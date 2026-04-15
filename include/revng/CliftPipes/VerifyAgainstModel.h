#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/CliftContainers.h"
#include "revng/PipeboxCommon/Helpers/PipeRuns/CliftFunctionMixin.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::piperuns {

class VerifyAgainstModel : public CliftFunctionMixin<VerifyAgainstModel> {
private:
  const model::Binary &Binary;

public:
  static constexpr llvm::StringRef Name = "verify-against-model";
  using Arguments = TypeList<PipeRunArgument<CliftFunctionContainer,
                                             "Modules",
                                             "function MLIR module(s)">>;

  VerifyAgainstModel(const class Model &Model,
                     llvm::StringRef StaticConfiguration,
                     llvm::StringRef Configuration,
                     CliftFunctionContainer &ModuleContainer) :
    CliftFunctionMixin(ModuleContainer), Binary(*Model.get().get()) {}

  void runOnCliftFunction(const model::Function &Function,
                          mlir::clift::FunctionOp MLIRFunction);
};

} // namespace revng::pypeline::piperuns
