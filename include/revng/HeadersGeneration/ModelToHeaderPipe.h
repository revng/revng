#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"

#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/MapContainer.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::pipes {

class ModelToHeader {
public:
  static constexpr llvm::StringRef Name = "model-to-header";
  static constexpr revng::pypeline::PipeArgumentDocumentation
    ArgumentsDocumentation[]{ { "Buffer", "" } };

  static void run(const Model &TheModel,
                  llvm::StringRef StaticConfig,
                  llvm::StringRef DynamicConfig,
                  BytesContainer &Buffer);
};

} // namespace revng::pypeline::pipes
