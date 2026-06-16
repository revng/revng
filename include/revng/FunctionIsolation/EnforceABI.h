#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/CollectCFG.h"
#include "revng/Model/NameBuilder.h"
#include "revng/PipeboxCommon/Helpers/PipeRuns/LLVMFunctionMixin.h"
#include "revng/PipeboxCommon/LLVMContainer.h"

namespace revng::pypeline::piperuns {

class EnforceABI : public LLVMFunctionMixin<EnforceABI> {
private:
  const model::Binary &Binary;
  const CFGMap &CFG;

public:
  static constexpr llvm::StringRef Name = "enforce-abi";
  using Arguments = TypeList<
    PipeRunArgument<const CFGMap, "CFG", "The per-function CFG data">,
    PipeRunArgument<LLVMFunctionContainer,
                    "Module",
                    "The LLVM Module(s) to run on">>;

  EnforceABI(const class Model &Model,
             llvm::StringRef Config,
             llvm::StringRef DynamicConfig,
             const CFGMap &CFG,
             LLVMFunctionContainer &Output) :
    LLVMFunctionMixin(Output), Binary(*Model.get().get()), CFG(CFG){};

  void runOnLLVMFunction(const model::Function &Function,
                         llvm::Function &LLVMFunction);
};

} // namespace revng::pypeline::piperuns
