#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/CliftContainers.h"
#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::piperuns {

class ImportTypes {
private:
  const model::Binary &Binary;
  CliftModuleContainer &Output;

public:
  static constexpr llvm::StringRef Name = "import-types";
  using Arguments = TypeList<PipeRunArgument<CliftModuleContainer,
                                             "Output",
                                             "Output MLIR container containing "
                                             "model type system",
                                             Access::Write>>;

  ImportTypes(const class Model &Model,
              llvm::StringRef Config,
              llvm::StringRef DynamicConfig,
              CliftModuleContainer &Output) :
    Binary(*Model.get().get()), Output(Output){};

  void run();
};

} // namespace revng::pypeline::piperuns
