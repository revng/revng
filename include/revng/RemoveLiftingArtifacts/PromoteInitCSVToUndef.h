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

class PromoteInitCSVToUndef : public LLVMFunctionMixin<PromoteInitCSVToUndef> {
private:
  const model::Binary &Binary;

public:
  static constexpr llvm::StringRef Name = "promote-init-csv-to-undef";
  using Arguments = SingleLLVMFunctionsArgument;

  PromoteInitCSVToUndef(const class Model &Model,
                        llvm::StringRef Config,
                        llvm::StringRef DynamicConfig,
                        LLVMFunctionContainer &ModuleContainer) :
    LLVMFunctionMixin(ModuleContainer), Binary(*Model.get().get()){};

  void runOnLLVMFunction(const model::Function &Function,
                         llvm::Function &LLVMFunction);
};

} // namespace revng::pypeline::piperuns
