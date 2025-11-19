#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/PTML/Tag.h"
#include "revng/PipeboxCommon/RawContainer.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipes/StringMap.h"
#include "revng/Yield/Pipes/ProcessAssembly.h"
#include "revng/Yield/Pipes/YieldControlFlow.h"

namespace revng::pipes {

class YieldAssembly {
public:
  static constexpr const auto Name = "yield-assembly";

public:
  inline std::array<pipeline::ContractGroup, 1> getContract() const {
    return { pipeline::ContractGroup(kinds::FunctionAssemblyInternal,
                                     0,
                                     kinds::FunctionAssemblyPTML,
                                     1,
                                     pipeline::InputPreservation::Preserve) };
  }

public:
  void run(pipeline::ExecutionContext &Context,
           const FunctionAssemblyStringMap &Input,
           FunctionAssemblyPTMLStringMap &Output);
};

} // namespace revng::pipes

namespace revng::pypeline {

using AssemblyContainer = FunctionToBytesContainer<"AssemblyContainer",
                                                   "text/x.asm+ptml">;

namespace piperuns {

class YieldAssembly {
private:
  const model::Binary &Model;
  const AssemblyInternalContainer &Input;
  AssemblyContainer &Output;

  model::CNameBuilder NameBuilder;
  ptml::MarkupBuilder B;

public:
  static constexpr llvm::StringRef Name = "YieldAssembly";
  using Arguments = TypeList<PipeRunArgument<const AssemblyInternalContainer,
                                             "Input",
                                             "The internal disassembly data">,
                             PipeRunArgument<AssemblyContainer,
                                             "Output",
                                             "Per-function disassembly",
                                             Access::Write>>;

  YieldAssembly(const class Model &Model,
                llvm::StringRef Config,
                llvm::StringRef DynamicConfig,
                const AssemblyInternalContainer &Input,
                AssemblyContainer &Output) :
    Model(*Model.get().get()),
    Input(Input),
    Output(Output),
    NameBuilder(*Model.get().get()) {}

  void runOnFunction(const model::Function &TheFunction);
};

} // namespace piperuns

} // namespace revng::pypeline
