#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"

#include "revng/ADT/TypeList.h"
#include "revng/Pipebox/Containers.h"
#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/Model.h"
#include "revng/PipeboxCommon/RawContainer.h"

namespace revng::pypeline::piperuns {

class ModelToHeader {
private:
  const model::Binary &Binary;
  PTMLCBytesContainer &Buffer;

public:
  static constexpr llvm::StringRef Name = "legacy-model-to-header";
  using Arguments = TypeList<PipeRunArgument<PTMLCBytesContainer,
                                             "Buffer",
                                             "The output C header of the model",
                                             Access::Write>>;

  ModelToHeader(const Model &TheModel,
                llvm::StringRef StaticConfig,
                llvm::StringRef DynamicConfig,
                PTMLCBytesContainer &Buffer);

  void run();
};

} // namespace revng::pypeline::piperuns
