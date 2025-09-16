#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"

#include "revng/ADT/TypeList.h"
#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/Model.h"
#include "revng/PipeboxCommon/RawContainer.h"

namespace revng::pypeline::pipes {

class ModelToHeader {
public:
  static constexpr llvm::StringRef Name = "model-to-header";
  using ArgumentsDocumentation = TypeList<
    PipeArgumentDocumentation<"Buffer", "">>;

  static void run(const Model &TheModel,
                  llvm::StringRef StaticConfig,
                  llvm::StringRef DynamicConfig,
                  BytesContainer &Buffer);
};

} // namespace revng::pypeline::pipes
