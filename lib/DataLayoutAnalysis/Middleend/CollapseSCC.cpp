//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include <set>
#include <type_traits>
#include <vector>

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Progress.h"

#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/Support/Debug.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"

#include "DLAStep.h"
#include "RemoveBackedges.h"

using namespace llvm;

static Logger<> LogVerbose("dla-collapse-verbose");

namespace dla {

using LTSN = LayoutTypeSystemNode;
using GraphNodeT = LTSN *;
using InstanceNodeT = EdgeFilteredGraph<GraphNodeT, isInstanceEdge>;
using EqualityNodeT = EdgeFilteredGraph<GraphNodeT, isEqualityEdge>;
using InstanceOffset0EdgeT = EdgeFilteredGraph<GraphNodeT, isInstanceOff0>;
using NonPointerNodeT = EdgeFilteredGraph<GraphNodeT, isNotPointerEdge>;

static bool sizeGreater(const LTSN *LHS, const LTSN *RHS) {
  return LHS->Size > RHS->Size;
}

template<typename NodeT>
static bool collapseSCCs(LayoutTypeSystem &TS) {

  std::set<GraphNodeT> VisitedNodes;

  using scc_t = std::vector<GraphNodeT>;
  llvm::SmallVector<scc_t, 0> ToCollapse;

  // We cannot start just from the roots because we cannot exclude that there
  // are loops without entries.
  for (const auto &Node : llvm::nodes(&TS)) {
    revng_assert(Node != nullptr);
    revng_log(LogVerbose, "## Analyzing SCCs from  " << Node);
    if (VisitedNodes.contains(Node)) {
      revng_log(LogVerbose, "## Was already visited");
      continue;
    }

    llvm::scc_iterator<NodeT> I = llvm::scc_begin(NodeT(Node));
    llvm::scc_iterator<NodeT> E = llvm::scc_end(NodeT(Node));
    for (const auto &SCC : llvm::make_range(I, E)) {
      revng_assert(not SCC.empty());
      if (VisitedNodes.contains(SCC[0]))
        continue;
      VisitedNodes.insert(SCC.begin(), SCC.end());

      if (LogVerbose.isEnabled()) {
        revng_log(LogVerbose, "# SCC has " << SCC.size() << " elements:");
        size_t I = 0;
        for (const LTSN *N : SCC)
          revng_log(LogVerbose, "  - " << I << ": " << N);
        revng_log(LogVerbose, "# SCC End");
      }

      // If the SCC is larger than just a single point we track all the nodes
      // that compose it as EqualityVisitedNodes, and we then add the SCC to the
      // set of SCC ToCollapse.
      if (SCC.size() > 1)
        ToCollapse.push_back(SCC);
    }
  }
  revng_log(LogVerbose, "## Collapsing " << ToCollapse.size() << " SCCs");

  for (scc_t &SCC : ToCollapse) {
    revng_log(LogVerbose, "# SCC with " << SCC.size() << " elements");
    revng_assert(not SCC.empty());
    // Reorder the SCC so that nodes with the same size are adjacent.
    // Nodes with larger sizes come first.
    llvm::stable_sort(SCC, sizeGreater);

    // Now, merge all the nodes that have the same size, while noting the
    // leaders.
    llvm::SmallSetVector<LayoutTypeSystemNode *, 8> Leaders;
    {
      auto LeaderIt = SCC.begin();
      auto CurrentSize = (*LeaderIt)->Size;
      auto End = SCC.end();

      while (LeaderIt != End) {

        Leaders.insert(*LeaderIt);

        // Find the first node with a size different from CurrentSize.
        auto EndCurrentSize = std::next(LeaderIt);
        while (EndCurrentSize != End and (*EndCurrentSize)->Size == CurrentSize)
          ++EndCurrentSize;

        // Merge all the nodes with the same size.
        TS.mergeNodes({ &*LeaderIt, &*EndCurrentSize });

        // Go to the next size.
        LeaderIt = EndCurrentSize;
      }
    }

    // Now, if the last Leader has Size 0, we merge it with the first Leader,
    // which represents the largest type.
    if (Leaders.size() > 1 and Leaders.back()->Size == 0) {
      TS.mergeNodes(llvm::SmallVector{ Leaders.front(), Leaders.back() });
      Leaders.pop_back();
    }

    // If a single leader is left, we're done, all the nodes have been merged
    // together.
    if (Leaders.size() < 2)
      continue;

    // If we're left with more than one leader, we just want to remove all edges
    // between them and replace them with offset-at-0 edges, from the larger
    // leader to the smaller.
    auto LeaderIt = Leaders.begin();
    auto LeaderEnd = Leaders.end();
    for (; LeaderIt != LeaderEnd; ++LeaderIt) {

      // For each leader we erase all the non-pointer edges going to other
      // leaders.
      auto *Leader = *LeaderIt;
      using DLAGraph = llvm::GraphTraits<LayoutTypeSystemNode *>;
      auto EdgeIt = DLAGraph::child_edge_begin(Leader);
      auto EdgeEnd = DLAGraph::child_edge_end(Leader);
      auto EdgeNext = DLAGraph::child_edge_end(Leader);

      for (; EdgeIt != EdgeEnd; EdgeIt = EdgeNext) {

        EdgeNext = std::next(EdgeIt);
        auto &Edge = *EdgeIt;

        // Skip pointers
        if (isPointerEdge(Edge))
          continue;

        const auto &[Child, Tag] = Edge;

        // Skip edges that go to non-leaders
        if (not Leaders.contains(Child))
          continue;

        TS.eraseEdge(Leader, EdgeIt);
      }

      // Then, for each leader, we add an instance-at-offset-0 edge to the next
      // leader, which is the one with immediately lower size.
      if (auto NextLeaderIt = std::next(LeaderIt); NextLeaderIt != LeaderEnd)
        TS.addInstanceLink(*LeaderIt, *NextLeaderIt, dla::OffsetExpression{});
    }
  }

  return ToCollapse.size();
}

static bool collapseEqualitySCC(LayoutTypeSystem &TS) {
  return collapseSCCs<EqualityNodeT>(TS);
}

static bool collapseInstanceAtOffset0SCC(LayoutTypeSystem &TS) {
  return collapseSCCs<InstanceOffset0EdgeT>(TS);
}

bool CollapseEqualitySCC::runOnTypeSystem(LayoutTypeSystem &TS) {

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyConsistency());

  revng_log(LogVerbose, "#### Collapsing Equality SCC: ... ");
  bool Changed = collapseEqualitySCC(TS);
  revng_log(LogVerbose, "#### Collapsing Equality SCC: Done!");

  if (VerifyLog.isEnabled()) {
    revng_assert(TS.verifyConsistency());
    revng_assert(TS.verifyNoEquality());
  }

  return Changed;
}

bool CollapseInstanceAtOffset0SCC::runOnTypeSystem(LayoutTypeSystem &TS) {
  Task T(2, "runOnTypeSystem");

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyConsistency());

  T.advance("collapseInstanceAtOffset0SCC");
  revng_log(LogVerbose, "#### Collapsing Instance-at-offset-0 SCC: ... ");
  bool Changed = collapseInstanceAtOffset0SCC(TS);
  revng_log(LogVerbose, "#### Collapsing Instance-at-offset-0 SCC: Done!");

  if (VerifyLog.isEnabled()) {
    revng_assert(TS.verifyConsistency());
    revng_assert(TS.verifyInstanceAtOffset0DAG());
  }

  T.advance("removeInstanceBackedgesFromInstanceAtOffset0Loops");
  Changed |= removeInstanceBackedgesFromInstanceAtOffset0Loops(TS);

  if (VerifyLog.isEnabled()) {
    revng_assert(TS.verifyConsistency());
    revng_assert(TS.verifyInstanceAtOffset0DAG());
    revng_assert(TS.verifyInstanceDAG());
  }

  return Changed;
}

} // end namespace dla
