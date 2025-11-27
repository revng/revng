#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::analyses {

class ConvertFunctionsToCABI {
public:
  static constexpr llvm::StringRef Name = "convert-functions-to-cabi";

  llvm::Error
  run(Model &Model, const Request &Incoming, llvm::StringRef Configuration);
};

} // namespace revng::pypeline::analyses
