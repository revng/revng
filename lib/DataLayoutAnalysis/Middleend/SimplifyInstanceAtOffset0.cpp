//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include <iostream>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"

#include "DLAStep.h"

using InstanceT = EdgeFilteredGraph<dla::LayoutTypeSystemNode *,
                                    dla::isInstanceEdge>;

using InstanceZeroT = EdgeFilteredGraph<dla::LayoutTypeSystemNode *,
                                        dla::isInstanceOff0>;

using CInstanceT = EdgeFilteredGraph<const dla::LayoutTypeSystemNode *,
                                     dla::isInstanceEdge>;

using CPointerT = EdgeFilteredGraph<const dla::LayoutTypeSystemNode *,
                                    dla::isPointerEdge>;

namespace dla {

static bool hasOutgoingPointer(const LayoutTypeSystemNode *N) {
  using PointerGraph = llvm::GraphTraits<CPointerT>;
  auto It = PointerGraph::child_begin(N);
  auto End = PointerGraph::child_end(N);
  return It != End;
};

static bool canBeCollapsed(const LayoutTypeSystemNode *Node,
                           const LayoutTypeSystemNode *Child) {
  // If Child has size or has an outgoing pointer edge, it can never be
  // collapsed.
  if (Child->Size or hasOutgoingPointer(Child))
    return false;

  llvm::SmallPtrSet<const LayoutTypeSystemNode *, 8> ChildrenOfChild;
  for (const auto *C : llvm::children<CInstanceT>(Child))
    ChildrenOfChild.insert(C);

  // If we find a successor of Node (different from Child at offset 0) such that
  // it can reach Child, then the instance-0 from Node to Child cannot be
  // collapsed, because that would introduce a loop.

  // First check the cheapest thing, i.e. if there is another edge to Child that
  // is not at offset 0. That is cheap to check and would already mean that we
  // cannot collapse, because collapsing the instance-0 edge would turn the
  // other edge to Child into a self-loop.
  for (const auto &Link : llvm::children_edges<CInstanceT>(Node)) {
    const auto &[OtherChild, EdgeTag] = Link;
    if (Child == OtherChild and not isInstanceOff0(Link))
      return false;
  }

  for (const auto &Link : llvm::children_edges<CInstanceT>(Node)) {
    const auto &[OtherChild, EdgeTag] = Link;

    revng_assert(OtherChild != Child or isInstanceOff0(Link));

    // Ignore the edge that would be collapsed.
    if (OtherChild == Child and isInstanceOff0(Link))
      continue;

    auto DFIt = llvm::df_begin(CInstanceT(OtherChild));
    auto DFEnd = llvm::df_end(CInstanceT(OtherChild));
    while (DFIt != DFEnd) {
      const auto *N = *DFIt;
      // Skip this DFS path early if we've reached a children of Child without
      // passing from Child.
      if (ChildrenOfChild.contains(N)) {
        DFIt.skipChildren();
        continue;
      }

      if (N == Child)
        return false;

      ++DFIt;
    }
  }

  // In all the other cases we can collapse.
  return true;
}

bool SimplifyInstanceAtOffset0::runOnTypeSystem(LayoutTypeSystem &TS) {
  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyInstanceDAG());

  bool Changed = false;

  for (LayoutTypeSystemNode *Root : llvm::nodes(&TS)) {
    if (not isInstanceRoot(Root))
      continue;

    // Pre-compute the post-order, since we're going to make changes to the
    // graphs while iterating on it, so we could invalidate post_order
    // iterators.
    llvm::SmallVector<LayoutTypeSystemNode *, 16> PostOrder;
    for (LayoutTypeSystemNode *Node : llvm::post_order(InstanceT(Root)))
      PostOrder.push_back(Node);

    for (LayoutTypeSystemNode *Node : PostOrder) {
      using InstanceZeroGraph = llvm::GraphTraits<InstanceZeroT>;
      auto It = InstanceZeroGraph::child_begin(Node);
      auto End = InstanceZeroGraph::child_end(Node);
      while (It != End) {
        // Check if it can be collapsed
        LayoutTypeSystemNode *ChildToCollapse = nullptr;
        if (canBeCollapsed(Node, *It))
          ChildToCollapse = *It;

        // Increment the iterator, so we don't invalidate when merging the
        // ChildToCollapse
        ++It;

        // If necessary, collapse the child into the parent
        if (ChildToCollapse)
          TS.mergeNodes({ Node, ChildToCollapse });
      }
    }
  }

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyInstanceDAG());

  return Changed;
}

} // end namespace dla
