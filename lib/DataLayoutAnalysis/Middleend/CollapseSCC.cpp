//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include <set>
#include <type_traits>
#include <vector>

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SCCIterator.h"

#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/Support/Debug.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"

#include "DLAStep.h"
#include "RemoveBackedges.h"

using namespace llvm;

static Logger<> LogVerbose("dla-collapse-verbose");
static Logger<> Log("dla-collapse");

namespace dla {

using LTSN = LayoutTypeSystemNode;
using GraphNodeT = LTSN *;
using InstanceNodeT = EdgeFilteredGraph<GraphNodeT, isInstanceEdge>;
using EqualityNodeT = EdgeFilteredGraph<GraphNodeT, isEqualityEdge>;
using InstanceOffset0EdgeT = EdgeFilteredGraph<GraphNodeT, isInstanceOff0>;
using NonPointerNodeT = EdgeFilteredGraph<GraphNodeT, isNotPointerEdge>;

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
    TS.mergeNodes(SCC);
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

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyConsistency());

  revng_log(LogVerbose, "#### Collapsing Instance-at-offset-0 SCC: ... ");
  bool Changed = collapseInstanceAtOffset0SCC(TS);
  revng_log(LogVerbose, "#### Collapsing Instance-at-offset-0 SCC: Done!");

  if (VerifyLog.isEnabled()) {
    revng_assert(TS.verifyConsistency());
    revng_assert(TS.verifyInstanceAtOffset0DAG());
  }

  Changed |= removeInstanceBackedgesFromInstanceAtOffset0Loops(TS);

  if (VerifyLog.isEnabled()) {
    revng_assert(TS.verifyConsistency());
    revng_assert(TS.verifyInstanceAtOffset0DAG());
    revng_assert(TS.verifyInstanceDAG());
  }

  return Changed;
}

} // end namespace dla
