#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Helpers/PipeRuns/LLVMFunctionMixin.h"
#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::piperuns {

class PromoteCSVs : public LLVMFunctionMixin<PromoteCSVs> {
private:
  const model::Binary &Binary;

public:
  static constexpr llvm::StringRef Name = "promote-csvs";
  using Arguments = TypeList<PipeRunArgument<LLVMFunctionContainer,
                                             "Module",
                                             "The LLVM Module(s) where the CSV "
                                             "will be promoted">>;

  PromoteCSVs(const class Model &Model,
              llvm::StringRef Config,
              llvm::StringRef DynamicConfig,
              LLVMFunctionContainer &ModuleContainer) :
    LLVMFunctionMixin(ModuleContainer), Binary(*Model.get().get()) {}

  void runOnLLVMFunction(const model::Function &Function,
                         llvm::Function &LLVMFunction);
};

} // namespace revng::pypeline::piperuns
