#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/NameBuilder.h"
#include "revng/Pipebox/Helpers.h"
#include "revng/PipeboxCommon/Helpers/PipeRuns/LLVMFunctionMixin.h"
#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::piperuns {

class RemoveLiftingArtifacts
  : public LLVMFunctionMixin<RemoveLiftingArtifacts> {
public:
  static constexpr llvm::StringRef Name = "remove-lifting-artifacts";
  using Arguments = SingleLLVMFunctionsArgument;

  RemoveLiftingArtifacts(const class Model &Model,
                         llvm::StringRef Config,
                         llvm::StringRef DynamicConfig,
                         LLVMFunctionContainer &ModuleContainer) :
    LLVMFunctionMixin(ModuleContainer) {}

  void runOnLLVMFunction(const model::Function &Function,
                         llvm::Function &LLVMFunction);
};

} // namespace revng::pypeline::piperuns
