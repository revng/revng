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
#include "revng/Yield/Pipes/YieldControlFlow.h"
#include "revng/Yield/SVG.h"

namespace revng::pipes {

void YieldControlFlow::run(pipeline::Context &Context,
                           const FunctionAssemblyStringMap &Input,
                           FunctionControlFlowStringMap &Output) {
  // Access the model
  const auto &Model = revng::getModelFromContext(Context);

  for (auto [Address, S] : Input) {
    auto MaybeFunction = TupleTree<yield::Function>::deserialize(S);
    revng_assert(MaybeFunction && MaybeFunction->verify());
    revng_assert((*MaybeFunction)->Entry() == Address);

    Output.insert_or_assign((*MaybeFunction)->Entry(),
                            yield::svg::controlFlowGraph(**MaybeFunction,
                                                         *Model));
  }
}

void YieldControlFlow::print(const pipeline::Context &,
                             llvm::raw_ostream &OS,
                             llvm::ArrayRef<std::string> Files) const {
  OS << *revng::ResourceFinder.findFile("bin/revng") << " magic ^_^\n";
}

} // end namespace revng::pipes

using namespace revng::pipes;
static RegisterFunctionStringMap<FunctionControlFlowStringMap> GraphContainer;

static pipeline::RegisterPipe<revng::pipes::YieldControlFlow> CFGPipe;
