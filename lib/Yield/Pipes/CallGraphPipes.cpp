//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/ControlFlowGraph.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/Model/Binary.h"
#include "revng/Ranks/Location.h"
#include "revng/TupleTree/TupleTree.h"
#include "revng/Yield/CrossRelations/CrossRelations.h"
#include "revng/Yield/Generated/ForwardDecls.h"
#include "revng/Yield/Pipes/ProcessCallGraph.h"
#include "revng/Yield/Pipes/YieldCallGraph.h"
#include "revng/Yield/Pipes/YieldCallGraphSlice.h"
#include "revng/Yield/SVG.h"

namespace revng::pypeline::piperuns {

void ProcessCallGraph::run() {
  using namespace yield::crossrelations;

  SortedVector<efa::ControlFlowGraph> Metadata;
  for (const ObjectID &Object : Input.objects())
    Metadata.insert(*Input.getElement(Object));

  *Output.getElement(ObjectID()) = CrossRelations(Metadata, Binary);
}

void YieldCallGraph::run() {
  using namespace yield::crossrelations;
  const TupleTree<CrossRelations> &Relations = Input.getElement(ObjectID());

  ptml::MarkupBuilder B;
  auto OS = Output.getOStream(ObjectID());
  // Convert the graph to SVG.
  *OS << yield::svg::callGraph(B, *Relations.get(), Binary);
}

void YieldCallGraphSlice::runOnFunction(const model::Function &Function) {
  using namespace yield::crossrelations;
  const TupleTree<CrossRelations> &Relations = Input.getElement(ObjectID());

  // Slice the graph for the current function and convert it to SVG
  auto SlicePoint = pipeline::locationString(revng::ranks::Function,
                                             Function.Entry());

  auto OS = Output.getOStream(ObjectID(Function.Entry()));
  *OS << yield::svg::callGraphSlice(B, SlicePoint, *Relations.get(), Binary);
}

} // namespace revng::pypeline::piperuns
