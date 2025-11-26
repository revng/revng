#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::analyses {

class DetectStackSize {
public:
  static constexpr llvm::StringRef Name = "detect-stack-size";

  llvm::Error run(Model &Model,
                  const Request &Incoming,
                  llvm::StringRef Configuration,
                  LLVMFunctionContainer &ModuleContainer);
};

} // namespace revng::pypeline::analyses
