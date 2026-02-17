#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::analyses {

class ImportFromC {
public:
  static constexpr llvm::StringRef Name = "import-from-c";

  llvm::Error
  run(Model &Model, const Request &Incoming, llvm::StringRef Configuration);
};

} // namespace revng::pypeline::analyses
