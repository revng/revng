#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipebox/Helpers.h"
#include "revng/PipeboxCommon/Helpers/PipeRuns/LLVMFunctionMixin.h"
#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::piperuns {

class SwitchToStatements : public LLVMFunctionMixin<SwitchToStatements> {
private:
  const TupleTree<model::Binary> &Model;

public:
  static constexpr llvm::StringRef Name = "switch-to-statements";
  using Arguments = SingleLLVMFunctionsArgument;

  SwitchToStatements(const class Model &Model,
                     llvm::StringRef Config,
                     llvm::StringRef DynamicConfig,
                     LLVMFunctionContainer &ModuleContainer) :
    LLVMFunctionMixin(ModuleContainer), Model(Model.get()) {}

  void runOnLLVMFunction(const model::Function &Function,
                         llvm::Function &LLVMFunction);
};

} // namespace revng::pypeline::piperuns
