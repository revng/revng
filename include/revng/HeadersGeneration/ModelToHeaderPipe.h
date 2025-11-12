#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"

#include "revng/ADT/TypeList.h"
#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/Model.h"
#include "revng/PipeboxCommon/RawContainer.h"

namespace revng::pypeline {

using CBytesContainer = BytesContainer<"CBytesContainer", "text/x.c+ptml">;

namespace piperuns {

class ModelToHeader {
public:
  static constexpr llvm::StringRef Name = "ModelToHeader";
  using Arguments = TypeList<
    PipeArgument<"Buffer", "The output C header of the model", Access::Write>>;

  static void run(const Model &TheModel,
                  llvm::StringRef StaticConfig,
                  llvm::StringRef DynamicConfig,
                  CBytesContainer &Buffer);
};

} // namespace piperuns

} // namespace revng::pypeline
