#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::analyses {

class ImportWellKnownModelsAnalysis {
public:
  static constexpr llvm::StringRef Name = "import-well-known-models";

  llvm::Error
  run(Model &Model, const Request &Incoming, llvm::StringRef Configuration);
};

} // namespace revng::pypeline::analyses
