#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/PipeboxCommon/LLVMContainer.h"

namespace revng::pypeline::piperuns {

class PromoteCSVs {
private:
  const model::Binary &Binary;
  LLVMFunctionContainer &ModuleContainer;

public:
  static constexpr llvm::StringRef Name = "PromoteCSVs";
  using Arguments = TypeList<PipeRunArgument<LLVMFunctionContainer,
                                             "Module",
                                             "The LLVM Module(s) where the CSV "
                                             "will be promoted">>;

  PromoteCSVs(const class Model &Model,
              llvm::StringRef Config,
              llvm::StringRef DynamicConfig,
              LLVMFunctionContainer &ModuleContainer) :
    Binary(*Model.get().get()), ModuleContainer(ModuleContainer) {}

  void runOnFunction(const model::Function &TheFunction);
};

} // namespace revng::pypeline::piperuns
