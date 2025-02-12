//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "DLAStep.h"
#include "FieldSizeComputation.h"

namespace dla {

bool DecomposeStridedEdges::runOnTypeSystem(LayoutTypeSystem &TS) {
  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());

  bool Changed = false;

  for (LayoutTypeSystemNode *Parent : llvm::nodes(&TS)) {

    using DLAGraph = llvm::GraphTraits<LayoutTypeSystemNode *>;
    auto EdgeIt = DLAGraph::child_edge_begin(Parent);
    auto EdgeEnd = DLAGraph::child_edge_end(Parent);
    auto EdgeNext = DLAGraph::child_edge_end(Parent);

    for (; EdgeIt != EdgeEnd; EdgeIt = EdgeNext) {
      EdgeNext = std::next(EdgeIt);

      auto &Edge = *EdgeIt;
      if (not isInstanceEdge(Edge))
        continue;

      const auto &[Child, Tag] = Edge;
      const auto &OffsetExpr = Tag->getOffsetExpr();
      auto NLayers = OffsetExpr.Strides.size();
      if (NLayers < 2)
        continue;

      Changed = true;

      // Setup the chain of nodes across which we will build the new chain of
      // single-layered strided edges.
      llvm::SmallVector<LayoutTypeSystemNode *> NodeChain;
      {
        NodeChain.reserve(NLayers + 1);
        // The first node of the chain is the original child.
        NodeChain.push_back(Child);
        // Then we have to insert NLayers - 1 new artificial node.
        const auto &NewNodes = TS.createArtificialLayoutTypes(NLayers - 1);
        for (auto *New : NewNodes)
          NodeChain.push_back(New);
        // The last node of the chain is the original parent.
        NodeChain.push_back(Parent);
      }

      // From outer to inner
      const auto &StridesTripCounts = llvm::zip_first(OffsetExpr.Strides,
                                                      OffsetExpr.TripCounts);
      // From inner to outer
      const auto &InToOut = llvm::reverse(StridesTripCounts);

      // Actually link the nodes in the chain with the new single-layered
      // strided edges
      for (const auto &Group : llvm::enumerate(InToOut)) {
        // Build the strided offset expression of the new edge
        OffsetExpression OE;
        // Take the Strides and TripCounts from the current layer.
        const auto &[S, TC] = Group.value();
        OE.Strides.push_back(S);
        OE.TripCounts.push_back(TC);
        // At the last iteration, representing the outermost array, copy the
        // offset as well (all the other iterations will have offset 0)
        if (Group.index() == NLayers - 1)
          OE.Offset = OffsetExpr.Offset;

        // Set up the predecessor and successor of the new single-layered
        // strided edge.
        auto Idx = Group.index();
        LayoutTypeSystemNode *Pred = NodeChain[Idx + 1];
        LayoutTypeSystemNode *Succ = NodeChain[Idx];
        // Link them
        auto &&[Tag, New] = TS.addInstanceLink(Pred, Succ, std::move(OE));
        if (Pred != Parent)
          Pred->Size = getFieldSize(Succ, Tag);
      }

      // Remove the old strided edge
      TS.eraseEdge(Parent, EdgeIt);
    }
  }

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());

  return Changed;
}

} // end namespace dla
