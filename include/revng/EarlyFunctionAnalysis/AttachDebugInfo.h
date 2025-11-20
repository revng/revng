#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/CollectCFG.h"
#include "revng/PipeboxCommon/Helpers/PipeRuns/LLVMFunctionMixin.h"
#include "revng/PipeboxCommon/LLVMContainer.h"

namespace revng::pypeline::piperuns {

class AttachDebugInfo : public LLVMFunctionMixin<AttachDebugInfo> {
private:
  const model::Binary &Binary;
  const CFGMap &CFG;

public:
  static constexpr llvm::StringRef Name = "attach-debug-info";
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
    LLVMFunctionMixin(ModuleContainer), Binary(*Model.get().get()), CFG(CFG){};

  void runOnLLVMFunction(const model::Function &Function,
                         llvm::Function &LLVMFunction);
};

} // namespace revng::pypeline::piperuns
