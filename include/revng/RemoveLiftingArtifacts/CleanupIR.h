#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::piperuns {

class CleanupIR {
private:
  LLVMRootContainer &ModuleContainer;

public:
  static constexpr llvm::StringRef Name = "cleanup-ir";
  using Arguments = TypeList<PipeRunArgument<LLVMRootContainer,
                                             "Module",
                                             "Merged llvm module with all "
                                             "functions">>;

  CleanupIR(const class Model &Model,
            llvm::StringRef StaticConfiguration,
            llvm::StringRef Configuration,
            LLVMRootContainer &ModuleContainer) :
    ModuleContainer(ModuleContainer){};

  void run();
};

} // namespace revng::pypeline::piperuns
