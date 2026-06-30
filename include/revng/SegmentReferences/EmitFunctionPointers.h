#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipebox/Helpers.h"
#include "revng/PipeboxCommon/Helpers/PipeRuns/LLVMFunctionMixin.h"
#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::piperuns {

class EmitFunctionPointers : public LLVMFunctionMixin<EmitFunctionPointers> {
private:
  const model::Binary &Binary;

public:
  static constexpr llvm::StringRef Name = "emit-function-pointers";
  using Arguments = SingleLLVMFunctionsArgument;

  EmitFunctionPointers(const class Model &Model,
                       llvm::StringRef Config,
                       llvm::StringRef DynamicConfig,
                       LLVMFunctionContainer &ModuleContainer) :
    LLVMFunctionMixin(ModuleContainer), Binary(*Model.get().get()) {}

  void runOnLLVMFunction(const model::Function &Function,
                         llvm::Function &LLVMFunction);
};

} // namespace revng::pypeline::piperuns
