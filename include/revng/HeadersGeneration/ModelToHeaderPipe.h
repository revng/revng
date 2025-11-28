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
private:
  const model::Binary &Binary;
  CBytesContainer &Buffer;

public:
  static constexpr llvm::StringRef Name = "model-to-header";
  using Arguments = TypeList<PipeRunArgument<CBytesContainer,
                                             "Buffer",
                                             "The output C header of the model",
                                             Access::Write>>;

  ModelToHeader(const Model &TheModel,
                llvm::StringRef StaticConfig,
                llvm::StringRef DynamicConfig,
                CBytesContainer &Buffer);

  void run();
};

} // namespace piperuns

} // namespace revng::pypeline
