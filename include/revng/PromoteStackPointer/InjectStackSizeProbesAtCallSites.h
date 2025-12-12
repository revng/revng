#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipebox/Helpers.h"
#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::piperuns {

struct InjectStackSizeProbesAtCallSites {
private:
  const model::Binary &Binary;
  LLVMFunctionContainer &ModuleContainer;

public:
  static constexpr llvm::StringRef Name = "inject-stack-size-probes-at-"
                                          "callsites";
  using Arguments = SingleLLVMFunctionsArgument;

  InjectStackSizeProbesAtCallSites(const class Model &Model,
                                   llvm::StringRef Config,
                                   llvm::StringRef DynamicConfig,
                                   LLVMFunctionContainer &ModuleContainer) :
    Binary(*Model.get().get()), ModuleContainer(ModuleContainer){};

  void runOnFunction(const model::Function &Function);
};

} // namespace revng::pypeline::piperuns
