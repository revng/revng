#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/CliftContainers.h"
#include "revng/PipeboxCommon/Helpers/PipeRuns/CliftFunctionMixin.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::piperuns {

class ImportFunctionDataModel
  : public CliftFunctionMixin<ImportFunctionDataModel> {

  const model::Binary &Binary;

public:
  static constexpr llvm::StringRef Name = "import-function-data-model";
  using Arguments = TypeList<PipeRunArgument<CliftFunctionContainer,
                                             "Modules",
                                             "function MLIR module(s)">>;

  explicit ImportFunctionDataModel(const Model &Model,
                                   llvm::StringRef Config,
                                   llvm::StringRef DynamicConfig,
                                   CliftFunctionContainer &ModuleContainer) :
    CliftFunctionMixin(ModuleContainer), Binary(*Model.get().get()) {}

  void runOnCliftFunction(const model::Function &Function,
                          clift::FunctionOp MLIRFunction);
};

class ImportDataModel {
  const model::Binary &Binary;
  CliftModuleContainer &TypesAndGlobals;

public:
  static constexpr llvm::StringRef Name = "import-data-model";
  using Arguments = TypeList<PipeRunArgument<CliftModuleContainer,
                                             "TypesAndGlobals",
                                             "Output MLIR container containing "
                                             "model type system",
                                             Access::ReadWrite>>;

  explicit ImportDataModel(const class Model &Model,
                           llvm::StringRef Config,
                           llvm::StringRef DynamicConfig,
                           CliftModuleContainer &TypesAndGlobals) :
    Binary(*Model.get().get()), TypesAndGlobals(TypesAndGlobals){};

  void run();
};

} // namespace revng::pypeline::piperuns
