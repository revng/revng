#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/CollectCFG.h"
#include "revng/Model/NameBuilder.h"
#include "revng/PipeboxCommon/LLVMContainer.h"

class EnforceABI;

namespace revng::pypeline::piperuns {

class EnforceABI {
private:
  const model::Binary &Binary;
  LLVMFunctionContainer &Output;
  const CFGMap &CFG;
  model::CNameBuilder NameBuilder;

public:
  static constexpr llvm::StringRef Name = "EnforceABI";
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
    Binary(*Model.get().get()), Output(Output), CFG(CFG), NameBuilder(Binary){};

  void runOnFunction(const model::Function &TheFunction);
};

} // namespace revng::pypeline::piperuns
