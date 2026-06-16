#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipebox/Containers.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline {

namespace piperuns {

class EmitCAsDirectory {
private:
  const model::Binary &Binary;
  const PTMLCBytesContainer &InputC;
  const PTMLCBytesContainer &InputTypesAndGlobals;
  const PTMLCBytesContainer &InputHelpers;
  RecompilableArchiveContainer &Output;

public:
  static constexpr llvm::StringRef Name = "emit-c-as-directory";
  using Arguments = TypeList<PipeRunArgument<const PTMLCBytesContainer,
                                             "DecompiledFunctions",
                                             "Input decompiled functions">,
                             PipeRunArgument<const PTMLCBytesContainer,
                                             "TypesAndGlobals",
                                             "Input type and global header">,
                             PipeRunArgument<const PTMLCBytesContainer,
                                             "Helpers",
                                             "Input helper header">,
                             PipeRunArgument<RecompilableArchiveContainer,
                                             "Output",
                                             "Output single archive containing "
                                             "everything needed for "
                                             "recompilation",
                                             Access::Write>>;

  EmitCAsDirectory(const Model &Model,
                   llvm::StringRef Config,
                   llvm::StringRef DynamicConfig,
                   const PTMLCBytesContainer &InputC,
                   const PTMLCBytesContainer &InputTypesAndGlobals,
                   const PTMLCBytesContainer &InputHelpers,
                   RecompilableArchiveContainer &Output) :
    Binary(*Model.get().get()),
    InputC(InputC),
    InputTypesAndGlobals(InputTypesAndGlobals),
    InputHelpers(InputHelpers),
    Output(Output) {}

  void run();
};

} // namespace piperuns

} // namespace revng::pypeline
