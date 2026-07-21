#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"

namespace revng::pypeline::analyses {

class DetectABI {
public:
  static constexpr llvm::StringRef Name = "detect-abi";

  llvm::Error run(Model &Model,
                  const Request &Incoming,
                  llvm::StringRef Configuration,
                  LLVMRootContainer &ModuleContainer);
};

} // namespace revng::pypeline::analyses
