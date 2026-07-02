#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/Pass/PassManager.h"

#include "revng/CliftPipes/Configuration.h"
#include "revng/Pipebox/Containers.h"
#include "revng/PipeboxCommon/CliftContainers.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::piperuns {

class EmitC {
private:
  CliftFunctionContainer &Input;
  PTMLCFunctionContainer &Output;

  CEmissionPipeConfiguration Configuration;

public:
  static constexpr llvm::StringRef Name = "emit-c";
  using Arguments = TypeList<PipeRunArgument<CliftFunctionContainer,
                                             "Modules",
                                             "function MLIR module(s)",
                                             Access::Read>,
                             PipeRunArgument<PTMLCFunctionContainer,
                                             "Output",
                                             "Decompiled per-function PTML-C",
                                             Access::Write>>;

  EmitC(const Model &Model,
        llvm::StringRef Config,
        llvm::StringRef DynamicConfig,
        CliftFunctionContainer &Input,
        PTMLCFunctionContainer &Output);

  void runOnFunction(const model::Function &Function);
};

} // namespace revng::pypeline::piperuns
