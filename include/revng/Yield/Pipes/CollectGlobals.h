#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/RawBinaryView.h"
#include "revng/Pipebox/Helpers.h"
#include "revng/PipeboxCommon/BinariesContainer.h"
#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/Yield/Pipes/ProcessCallGraph.h"

namespace revng::pypeline::piperuns {

/// Record global variables and the isolated functions referencing them.
class CollectGlobals {
private:
  const model::Binary &Binary;
  RawBinaryView BinaryView;
  LLVMFunctionContainer &Input;
  CrossRelationsContainer &Output;

public:
  static constexpr llvm::StringRef Name = "collect-globals";
  using Arguments = TypeList<PipeRunArgument<const BinariesContainer,
                                             "Binaries",
                                             "The input binaries, read for the "
                                             "text of the strings">,
                             PipeRunArgument<LLVMFunctionContainer,
                                             "Input",
                                             "Isolated functions to read "
                                             "segment references from",
                                             // The module is only read, but the
                                             // analysis it is handed to needs
                                             // it mutable.
                                             Access::Read>,
                             PipeRunArgument<CrossRelationsContainer,
                                             "Output",
                                             "Cross relations to record the "
                                             "global variables into",
                                             Access::ReadWrite>>;

  CollectGlobals(const class Model &Model,
                 llvm::StringRef Config,
                 llvm::StringRef DynamicConfig,
                 const BinariesContainer &Binaries,
                 LLVMFunctionContainer &Input,
                 CrossRelationsContainer &Output) :
    Binary(*Model.get().get()),
    BinaryView(makeBinaryView(Model, Binaries)),
    Input(Input),
    Output(Output) {}

  static llvm::Error checkPrecondition(const class Model &Model) {
    return RawBinaryView::checkPrecondition(*Model.get().get());
  }

  void run();
};

} // namespace revng::pypeline::piperuns
