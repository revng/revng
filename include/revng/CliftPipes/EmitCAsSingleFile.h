#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/CliftPipes/Configuration.h"
#include "revng/Pipebox/Containers.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline {

namespace piperuns {

class EmitCAsSingleFile {
private:
  const model::Binary &Binary;
  const PTMLCFunctionContainer &Input;
  PTMLCContainer &Output;

  CEmissionPipeConfiguration Configuration;

public:
  static constexpr llvm::StringRef Name = "emit-c-as-single-file";
  using Arguments = TypeList<PipeRunArgument<const PTMLCFunctionContainer,
                                             "DecompiledFunctions",
                                             "Input decompiled function">,
                             PipeRunArgument<PTMLCContainer,
                                             "Output",
                                             "Output single C+PTML",
                                             Access::Write>>;

  EmitCAsSingleFile(const Model &Model,
                    llvm::StringRef Config,
                    llvm::StringRef DynamicConfig,
                    const PTMLCFunctionContainer &Input,
                    PTMLCContainer &Output);

  void run();
};

} // namespace piperuns

} // namespace revng::pypeline
