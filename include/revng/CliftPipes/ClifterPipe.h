#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/CliftContainers.h"
#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::piperuns {

class Clifter {
private:
  const model::Binary &Binary;
  const LLVMFunctionContainer &Input;
  CliftFunctionContainer &Output;

public:
  static constexpr llvm::StringRef Name = "clifter";
  using Arguments = TypeList<PipeRunArgument<const LLVMFunctionContainer,
                                             "Input",
                                             "Input LLVM module(s) to be "
                                             "converted to Clift">,
                             PipeRunArgument<CliftFunctionContainer,
                                             "Output",
                                             "Output MLIR container(s) with "
                                             "Clift dialect",
                                             Access::Write>>;

  Clifter(const class Model &Model,
          llvm::StringRef Config,
          llvm::StringRef DynamicConfig,
          const LLVMFunctionContainer &Input,
          CliftFunctionContainer &Output);

  void runOnFunction(const model::Function &Function);
};

} // namespace revng::pypeline::piperuns
