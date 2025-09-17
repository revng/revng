#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/TypeList.h"
#include "revng/Model/NameBuilder.h"
#include "revng/PTML/Tag.h"
#include "revng/PipeboxCommon/Model.h"
#include "revng/PipeboxCommon/RawContainer.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipes/StringMap.h"
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

namespace revng::pypeline::pipes {

class YieldAssembly {
private:
  const Model &Model;
  const FunctionToBytesContainer &Input;
  FunctionToBytesContainer &Output;
  model::CNameBuilder NameBuilder;
  ptml::MarkupBuilder B;

public:
  static constexpr llvm::StringRef Name = "yield-assembly";
  using Arguments = TypeList<
    PipeArgument<const FunctionToBytesContainer, "Input", "">,
    PipeArgument<FunctionToBytesContainer, "Output", "">>;

  YieldAssembly(const class Model &Model,
                llvm::StringRef Config,
                llvm::StringRef DynamicConfig,
                const FunctionToBytesContainer &Input,
                FunctionToBytesContainer &Output) :
    Model(Model),
    Input(Input),
    Output(Output),
    NameBuilder(*Model.get().get()) {}

  void runOnFunction(const model::Function &TheFunction);
};

} // namespace revng::pypeline::pipes
