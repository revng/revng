#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"

#include "revng/Pypeline/Common.h"
#include "revng/Pypeline/MapContainer.h"
#include "revng/Pypeline/Model.h"

namespace revng::pypeline::pipes {

class ModelToHeader {
public:
  static constexpr llvm::StringRef Name = "model-to-header";

  ModelToHeader(llvm::StringRef Config){};

  ObjectDependencies run(const Model *TheModel,
                         Request Incoming,
                         Request Outgoing,
                         llvm::StringRef DynamicConfig,
                         RootBuffer &Buffer);
};

} // namespace revng::pypeline::pipes
