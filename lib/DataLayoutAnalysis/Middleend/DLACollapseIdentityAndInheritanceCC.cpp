//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include <set>
#include <type_traits>
#include <vector>

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SCCIterator.h"

#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/Support/Debug.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"

#include "../DLAHelpers.h"
#include "DLAStep.h"

using namespace llvm;

static Logger<> LogVerbose("dla-collapse-verbose");
static Logger<> Log("dla-collapse");

namespace dla {

using LTSN = LayoutTypeSystemNode;
using GraphNodeT = LTSN *;
using InstanceNodeT = EdgeFilteredGraph<GraphNodeT, isInstanceEdge>;
using InheritanceNodeT = EdgeFilteredGraph<GraphNodeT, isInheritanceEdge>;
using EqualityNodeT = EdgeFilteredGraph<GraphNodeT, isEqualityEdge>;
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
    if (VisitedNodes.count(Node)) {
      revng_log(LogVerbose, "## Was already visited");
      continue;
    }

    llvm::scc_iterator<NodeT> I = llvm::scc_begin(NodeT(NonPointerNodeT(Node)));
    llvm::scc_iterator<NodeT> E = llvm::scc_end(NodeT(NonPointerNodeT(Node)));
    for (const auto &SCC : llvm::make_range(I, E)) {
      revng_assert(not SCC.empty());
      if (VisitedNodes.count(SCC[0]))
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

static bool collapseInheritanceSCC(LayoutTypeSystem &TS) {
  using InheritanceGraph = EdgeFilteredGraph<NonPointerNodeT,
                                             isInheritanceEdge>;
  return collapseSCCs<InheritanceGraph>(TS);
}

static bool collapseEqualitySCC(LayoutTypeSystem &TS) {
  return collapseSCCs<EdgeFilteredGraph<NonPointerNodeT, isEqualityEdge>>(TS);
}

bool CollapseIdentityAndInheritanceCC::runOnTypeSystem(LayoutTypeSystem &TS) {
  if (Log.isEnabled())
    TS.dumpDotOnFile("before-collapse.dot");
  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyConsistency());

  revng_log(LogVerbose, "#### Merging Equality SCC: ... ");
  bool CollapsedEqual = collapseEqualitySCC(TS);
  revng_log(LogVerbose, "#### Merging Equality SCC: Done!");

  if (Log.isEnabled())
    TS.dumpDotOnFile("after-collapse-equality.dot");

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyNoEquality());

  revng_log(LogVerbose, "#### Merging Inheritance SCC: ... ");
  bool CollapsedInheritance = collapseInheritanceSCC(TS);
  revng_log(LogVerbose, "#### Merging Inheritance SCC: Done!");

  if (Log.isEnabled())
    TS.dumpDotOnFile("after-collapse-inheritance.dot");

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyInheritanceDAG());

  // The following assertion is not really necessary.
  // In principle all the rest of the pipeline should work fine without it, but
  // I think that if it's triggered something might be wrong in how we emit
  // instance edges earlier. If it's ever triggered, please double check.
  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyInstanceDAG());

  revng_log(LogVerbose, "#### Merging Mixed SCC: ... ");
  bool Removed = false;
  Removed = removeInstanceBackedgesFromInheritanceLoops(TS);
  revng_log(LogVerbose, "#### Merging Mixed SCC: Done!");

  if (Log.isEnabled())
    TS.dumpDotOnFile("after-collapse.dot");

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());

  return CollapsedEqual or CollapsedInheritance or Removed;
}

} // end namespace dla
