#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipebox/Containers.h"
#include "revng/PipeboxCommon/CliftContainers.h"
#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::piperuns {

class EmitTypeAndGlobalHeader {
private:
  const model::Binary &Binary;
  const CliftModuleContainer &Input;
  PTMLCBytesContainer &Output;

public:
  static constexpr llvm::StringRef Name = "emit-type-and-global-header";
  using Arguments = TypeList<PipeRunArgument<CliftModuleContainer,
                                             "Input",
                                             "MLIR container containing "
                                             "the type system, as well as "
                                             "function and segment "
                                             "declarations",
                                             Access::Read>,
                             PipeRunArgument<PTMLCBytesContainer,
                                             "Output",
                                             "The model header",
                                             Access::Write>>;

  EmitTypeAndGlobalHeader(const class Model &Model,
                          llvm::StringRef Config,
                          llvm::StringRef DynamicConfig,
                          const CliftModuleContainer &Input,
                          PTMLCBytesContainer &Output) :
    Binary(*Model.get().get()), Input(Input), Output(Output){};

  void run();
};

class EmitHelperHeader {
private:
  const model::Binary &Binary;
  const CliftFunctionContainer &Input;
  PTMLCBytesContainer &Output;

public:
  static constexpr llvm::StringRef Name = "emit-helper-header";
  using Arguments = TypeList<PipeRunArgument<CliftFunctionContainer,
                                             "Input",
                                             "MLIR container",
                                             Access::Read>,
                             PipeRunArgument<PTMLCBytesContainer,
                                             "Output",
                                             "The helper header",
                                             Access::Write>>;

  EmitHelperHeader(const class Model &Model,
                   llvm::StringRef Config,
                   llvm::StringRef DynamicConfig,
                   const CliftFunctionContainer &Input,
                   PTMLCBytesContainer &Output) :
    Binary(*Model.get().get()), Input(Input), Output(Output){};

  void run();
};

} // namespace revng::pypeline::piperuns
