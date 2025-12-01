#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/CliftContainer.h"
#include "revng/PipeboxCommon/Helpers/PipeRuns/CliftFunctionMixin.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::piperuns {

class ImportModelNames : public CliftFunctionMixin<ImportModelNames> {
private:
  const model::Binary &Binary;

public:
  static constexpr llvm::StringRef Name = "import-model-names";
  using Arguments = TypeList<PipeRunArgument<CliftFunctionContainer,
                                             "Modules",
                                             "function MLIR module(s)">>;

  ImportModelNames(const Model &Model,
                   llvm::StringRef Config,
                   llvm::StringRef DynamicConfig,
                   CliftFunctionContainer &ModuleContainer) :
    CliftFunctionMixin(ModuleContainer), Binary(*Model.get().get()) {}

  void runOnCliftFunction(const model::Function &Function,
                          mlir::clift::FunctionOp MLIRFunction);
};

} // namespace revng::pypeline::piperuns
