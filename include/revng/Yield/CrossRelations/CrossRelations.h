#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/GenericGraph.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraph.h"
#include "revng/Model/Binary.h"
#include "revng/Yield/CallGraphs/Graph.h"
#include "revng/Yield/CrossRelations/RelationDescription.h"

#include "revng/Yield/CrossRelations/Generated/Early/CrossRelations.h"

namespace yield::crossrelations {

class CrossRelations : public generated::CrossRelations {
private:
  using Node = BidirectionalNode<std::string>;

public:
  CrossRelations() = default;
  CrossRelations(const SortedVector<efa::ControlFlowGraph> &Metadata,
                 const model::Binary &Binary);

  GenericGraph<Node, 16, true> toCallGraph() const;
  yield::calls::PreLayoutGraph toYieldGraph() const;
};

} // namespace yield::crossrelations

#include "revng/Yield/CrossRelations/Generated/Late/CrossRelations.h"
