//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include <memory>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/GenericDomTree.h"
#include "llvm/Support/GenericDomTreeConstruction.h"

#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/Support/Debug.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"

#include "../DLAHelpers.h"
#include "DLAStep.h"

using LTSN = dla::LayoutTypeSystemNode;
using GraphNodeT = LTSN *;
using CGraphNodeT = const LTSN *;

template<typename T>
using InheritanceView = EdgeFilteredGraph<T, dla::isInheritanceEdge>;
using PostDomTree = llvm::DominatorTreeOnView<LTSN, true, InheritanceView>;

namespace llvm {
namespace DomTreeBuilder {
template void Calculate<LTSN, true, InheritanceView>(PostDomTree &);
} // namespace DomTreeBuilder
} // end namespace llvm

static Logger<> Log("dla-inheritance-tree");

namespace dla {

using VecToCollapseT = std::vector<LTSN *>;
using VecToCollapseIter = VecToCollapseT::iterator;
using CollapseSetT = std::set<std::unique_ptr<VecToCollapseT>>;
using CollapseSetIt = CollapseSetT::iterator;

static bool IterCmp(const CollapseSetIt &A, const CollapseSetIt &B) {
  return &*A < &*B;
}

using CollapseSetIterCmp = std::integral_constant<decltype(IterCmp) &, IterCmp>;

bool MakeInheritanceTree::runOnTypeSystem(LayoutTypeSystem &TS) {
  bool Changed = false;
  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());

  if (Log.isEnabled())
    TS.dumpDotOnFile("before-make-inheritance-tree.dot");

  PostDomTree PDT;
  PDT.recalculate(TS);

  const auto hasAtMostOneInheritanceEdge = [](const LTSN *Node) {
    using CInheritanceNodeT = EdgeFilteredGraph<CGraphNodeT, isInheritanceEdge>;
    using CGT = llvm::GraphTraits<CInheritanceNodeT>;
    auto Beg = CGT::child_begin(Node);
    auto End = CGT::child_end(Node);
    return (Beg == End) or (std::next(Beg) == End);
  };

  CollapseSetT VecsToCollapse;
  std::map<LTSN *, CollapseSetIt> NodeToCollapsingSet;

  const auto CreateNewInVecsToCollapse = [&VecsToCollapse]() {
    auto Tmp = std::make_unique<VecToCollapseT>();
    return VecsToCollapse.insert(std::move(Tmp)).first;
  };

  for (LTSN *Node : llvm::nodes(&TS)) {
    // If Node has at most one outgoing inheritance edge, do nothing
    if (hasAtMostOneInheritanceEdge(Node))
      continue;

    // If Node is already part of a set of nodes to collapse, skip it
    if (NodeToCollapsingSet.count(Node))
      continue;
    // Here we're sure Node has at least two outgoing inheritance edges

    // Alloc the new vector of nodes to collapse.
    auto NewVecToCollapseIt = CreateNewInVecsToCollapse();
    using IterSet = llvm::SmallSet<CollapseSetIt, 8, CollapseSetIterCmp>;
    // While filling the new vector (V) of nodes to collapse, we might encounter
    // other nodes (N) that is already part of other vectors (W_N) of nodes to
    // collapse. If this happens, all the (W_N) must be merged with (V).
    IterSet IntersectingSetToCollapse;
    {
      // Post order visit, from Node, to its immediate post dominator.
      // All the nodes that are reachable between Node and PostDom will be
      // collapsed on the same node.
      // The PostDom might be a nullptr: in that case, no reachable node will
      // be equal to it.
      LTSN *PostDom = PDT.getNode(Node)->getIDom()->getBlock();
      std::set<LTSN *> PostDomSet = { PostDom };

      using InhNodeT = EdgeFilteredGraph<GraphNodeT, isInheritanceEdge>;
      for (LTSN *Reachable : llvm::post_order_ext(InhNodeT(Node), PostDomSet)) {
        // Prevent Node and PostDom to be collapsed with the other nodes.
        if (Reachable == Node or Reachable == PostDom)
          continue;

        auto It = NodeToCollapsingSet.find(Reachable);
        if (It != NodeToCollapsingSet.end()) {
          // Reachable was already found from another exploration and it's
          // already part of a set to collapse.
          IntersectingSetToCollapse.insert(It->second);
        } else {
          NewVecToCollapseIt->get()->push_back(Reachable);
        }
      }
    }
    revng_assert(not NewVecToCollapseIt->get()->empty()
                 or not IntersectingSetToCollapse.empty());

    if (not IntersectingSetToCollapse.empty()) {
      // Resize *NewVecToCollapseIt so that it's large enough to have space for
      // all the nodes in all the IntersectingSetToCollapse.
      VecToCollapseT &Into = *NewVecToCollapseIt->get();
      auto FinalSize = Into.size();
      for (CollapseSetIt It : IntersectingSetToCollapse)
        FinalSize += It->get()->size();
      NewVecToCollapseIt->get()->reserve(FinalSize);

      // Copy all the intersecting sets to collapse into the new set to
      // collapse, that encompasses all of them.
      for (CollapseSetIt It : IntersectingSetToCollapse) {
        VecToCollapseT &From = *It->get();
        Into.insert(Into.end(), From.begin(), From.end());
        VecsToCollapse.erase(It);
      }
    }

    for (LTSN *N : *NewVecToCollapseIt->get())
      NodeToCollapsingSet[N] = NewVecToCollapseIt;
  }

  for (auto &VecUPtr : VecsToCollapse) {
    if (VecUPtr.get()->size() > 1ULL) {
      if (Log.isEnabled()) {
        revng_log(Log, "Merging nodes to make Inheritance Tree -------");
        for (LTSN *N : *VecUPtr)
          revng_log(Log, "   " << N->ID);
      }
      TS.mergeNodes(*VecUPtr.get());
      Changed = true;
    }
  }

  if (Log.isEnabled())
    TS.dumpDotOnFile("after-make-inheritance-tree.dot");
  if (VerifyLog.isEnabled()) {
    revng_assert(TS.verifyInheritanceTree());
    revng_assert(TS.verifyInheritanceDAG());
  }

  if (Changed) {
    // Whenever we collapse nodes, we might end up creating loops of inheritance
    // nodes that are closed by a instance backedge. Remove them.
    removeInstanceBackedgesFromInheritanceLoops(TS);
  }

  if (Log.isEnabled())
    TS.dumpDotOnFile("after-final-inheritance-tree.dot");
  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyInheritanceTree() and TS.verifyDAG());

  return Changed;
}

} // end namespace dla
