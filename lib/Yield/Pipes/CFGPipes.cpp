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

using ptml::PTMLBuilder;

namespace revng::pipes {

void YieldControlFlow::run(pipeline::ExecutionContext &Context,
                           const FunctionAssemblyStringMap &Input,
                           FunctionControlFlowStringMap &Output) {
  // Access the model
  const auto &Model = revng::getModelFromContext(Context);
  ptml::PTMLBuilder B;

  for (const model::Function &Function :
       getFunctionsAndCommit(Context, Output.name())) {
    MetaAddress Address = Function.Entry();
    llvm::StringRef YamlText = Input.at(Address);
    auto MaybeFunction = TupleTree<yield::Function>::fromString(YamlText);

    revng_assert(MaybeFunction && MaybeFunction->verify());
    revng_assert((*MaybeFunction)->Entry() == Address);

    Output.insert_or_assign((*MaybeFunction)->Entry(),
                            yield::svg::controlFlowGraph(B,
                                                         **MaybeFunction,
                                                         *Model));
  }
}

} // end namespace revng::pipes

using namespace revng::pipes;
using namespace pipeline;
static RegisterDefaultConstructibleContainer<FunctionControlFlowStringMap>
  GraphContainer;

static pipeline::RegisterPipe<revng::pipes::YieldControlFlow> CFGPipe;
