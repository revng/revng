#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/CollectCFG.h"
#include "revng/PipeboxCommon/LLVMContainer.h"

namespace revng::pypeline::piperuns {

class AttachDebugInfo {
private:
  const model::Binary &Binary;
  const CFGMap &CFG;
  LLVMFunctionContainer &ModuleContainer;

public:
  static constexpr llvm::StringRef Name = "AttachDebugInfo";
  using Arguments = TypeList<
    PipeRunArgument<const CFGMap, "CFG", "Function control-flow data">,
    PipeRunArgument<LLVMFunctionContainer,
                    "Module",
                    "function LLVM module(s)">>;

  AttachDebugInfo(const class Model &Model,
                  llvm::StringRef Config,
                  llvm::StringRef DynamicConfig,
                  const CFGMap &CFG,
                  LLVMFunctionContainer &ModuleContainer) :
    Binary(*Model.get().get()), CFG(CFG), ModuleContainer(ModuleContainer){};

  void runOnFunction(const model::Function &TheFunction);
};

} // namespace revng::pypeline::piperuns
