//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Pipeline/Pipe.h"
#include "revng/Pipeline/RegisterContainerFactory.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Yield/Function.h"
#include "revng/Yield/Pipes/YieldControlFlowPipe.h"
#include "revng/Yield/SVG.h"

namespace revng::pipes {

void YieldControlFlowPipe::run(pipeline::Context &Context,
                               const FunctionStringMap &Input,
                               FunctionStringMap &Output) {
  // Access the model
  const auto &Model = revng::pipes::getModelFromContext(Context);

  for (auto [Address, S] : Input) {
    auto MaybeFunction = TupleTree<yield::Function>::deserialize(S);
    revng_assert(MaybeFunction && MaybeFunction->verify());
    revng_assert((*MaybeFunction)->Entry == Address);

    Output.insert_or_assign((*MaybeFunction)->Entry,
                            yield::svg::controlFlow(**MaybeFunction, *Model));
  }
}

void YieldControlFlowPipe::print(const pipeline::Context &,
                                 llvm::raw_ostream &OS,
                                 llvm::ArrayRef<std::string> Files) const {
  OS << *revng::ResourceFinder.findFile("bin/revng") << " magic ^_^\n";
}

std::array<pipeline::ContractGroup, 1>
YieldControlFlowPipe::getContract() const {
  return { pipeline::ContractGroup(FunctionAssemblyInternal,
                                   pipeline::Exactness::Exact,
                                   0,
                                   FunctionControlFlowGraphSVG,
                                   1,
                                   pipeline::InputPreservation::Preserve) };
}

} // end namespace revng::pipes

static revng::pipes::RegisterFunctionStringMap
  GraphContainer("FunctionControlFlowGraphSVG",
                 "application/x.yaml.cfg.svg-body",
                 revng::pipes::FunctionControlFlowGraphSVG);

static pipeline::RegisterPipe<revng::pipes::YieldControlFlowPipe> CFGPipe;
