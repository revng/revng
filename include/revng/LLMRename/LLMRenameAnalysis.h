#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipebox/Containers.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::analyses {

class LLMRename {
public:
  static constexpr llvm::StringRef Name = "llm-rename";

  llvm::Error run(Model &Model,
                  const Request &Incoming,
                  llvm::StringRef Configuration,
                  const PTMLCFunctionContainer &Input);

  bool isAvailable() const;
};

} // namespace revng::pypeline::analyses
