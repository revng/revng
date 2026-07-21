//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Yield/Function.h"
#include "revng/Yield/Pipes/YieldControlFlow.h"
#include "revng/Yield/SVG.h"

namespace revng::pypeline::piperuns {

void YieldCFG::runOnFunction(const model::Function &Function) {
  ObjectID Object(Function.Entry());
  auto YieldFunction = Input.getElement(Object);

  revng_assert(YieldFunction->verify());
  revng_assert(YieldFunction->Entry() == Function.Entry());

  auto OS = Output.getOStream(Object);
  *OS << yield::svg::controlFlowGraph(B, *YieldFunction, Binary);
}

} // namespace revng::pypeline::piperuns
