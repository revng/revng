#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/CliftContainers.h"
#include "revng/PipeboxCommon/Helpers/PipeRuns/CliftFunctionMixin.h"
#include "revng/PipeboxCommon/Model.h"
#include "revng/Support/CDialect.h"

namespace revng::pypeline::piperuns {

class ConfigureFunctionCDialect
  : public CliftFunctionMixin<ConfigureFunctionCDialect> {

  CDialect Dialect;

public:
  static constexpr llvm::StringRef Name = "configure-function-c-dialect";
  using Arguments = TypeList<PipeRunArgument<CliftFunctionContainer,
                                             "Modules",
                                             "function MLIR module(s)">>;

  explicit ConfigureFunctionCDialect(const Model &Model,
                                     llvm::StringRef Config,
                                     llvm::StringRef DynamicConfig,
                                     CliftFunctionContainer &ModuleContainer);

  void runOnCliftFunction(const model::Function &Function,
                          clift::FunctionOp MLIRFunction);
};

} // namespace revng::pypeline::piperuns
