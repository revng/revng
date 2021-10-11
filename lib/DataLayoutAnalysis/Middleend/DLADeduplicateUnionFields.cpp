//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include <algorithm>
#include <compare>
#include <cstdint>
#include <iterator>

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"

#include "../DLAHelpers.h"
#include "DLAStep.h"

using LTSN = dla::LayoutTypeSystemNode;
using order = std::strong_ordering;
using Link = dla::LayoutTypeSystemNode::Link;
using EdgeList = std::vector<Link>;
using Tag = dla::TypeLinkTag;

using namespace llvm;

static Logger<> Log("dla-deduplicate-union-fields");
static Logger<> CmpLog("dla-duf-comparisons");

namespace dla {

///\brief Strong ordering for nodes: order by size, then by number of successors
static order cmpNodes(const LTSN *A, const LTSN *B) {
  if (A == B)
    return order::equal;

  if (not A)
    return order::less;

  if (not B)
    return order::greater;

  const auto SizeCmp = A->Size <=> B->Size;
  if (SizeCmp != order::equal) {
    revng_log(CmpLog, "Different sizes");
    return SizeCmp;
  }

  size_t NChild1 = A->Successors.size();
  size_t NChild2 = B->Successors.size();
  const auto NChildCmp = NChild1 <=> NChild2;
  if (NChildCmp != order::equal) {
    revng_log(CmpLog,
              "Different number of successors:  node "
                << A->ID << " has " << NChild1 << " successors, node " << B->ID
                << " has" << NChild2 << " successors");
    return NChildCmp;
  }

  // TODO: check pointer edges
  return order::equal;
}

///\brief Strong ordering for edges: order by kind, then by offset expression
///
///\note Inheritance and instance at offset 0 can be considered equivalent when
/// comparing subtrees.
static order
cmpEdgeTags(const Tag *A, const Tag *B, bool IgnoreInheritance = true) {
  if (A == B)
    return order::equal;
  revng_assert(A != nullptr and B != nullptr);

  // If A is an inheritance edge, consider it as an instance-offset-0 edge
  auto KindA = A->getKind();
  OffsetExpression OffA;
  if (KindA == TypeLinkTag::LK_Inheritance) {
    if (IgnoreInheritance)
      KindA = TypeLinkTag::LK_Instance;
    OffA.Offset = 0;
  } else {
    OffA = A->getOffsetExpr();
  }

  // If B is an inheritance edge, consider it as an instance-offset-0 edge
  auto KindB = B->getKind();
  OffsetExpression OffB;
  if (KindB == TypeLinkTag::LK_Inheritance) {
    if (IgnoreInheritance)
      KindB = TypeLinkTag::LK_Instance;
    OffB.Offset = 0;
  } else {
    OffB = B->getOffsetExpr();
  }

  const auto KindCmp = KindA <=> KindB;
  if (KindCmp != order::equal)
    return KindCmp;

  const auto OffsetCmp = OffA.Offset <=> OffB.Offset;
  if (OffsetCmp != order::equal)
    return OffsetCmp;

  const auto StrideSizeCmp = OffA.Strides.size() <=> OffB.Strides.size();
  if (StrideSizeCmp != order::equal)
    return StrideSizeCmp;
  for (const auto &[StrA, StrB] : llvm::zip(OffA.Strides, OffB.Strides)) {
    const auto StrideCmp = StrA <=> StrB;
    if (StrideCmp != order::equal)
      return StrideCmp;
  }

  const auto TCSizeCmp = OffA.TripCounts.size() <=> OffB.TripCounts.size();
  if (TCSizeCmp != order::equal)
    return TCSizeCmp;
  for (const auto &[TCA, TCB] : llvm::zip(OffA.TripCounts, OffB.TripCounts)) {
    if (not TCA and not TCB)
      continue;
    if (TCA and not TCB)
      return order::less;
    if (not TCA and TCB)
      return order::greater;

    if (TCA and TCB) {
      const auto TCCmp = *TCA <=> *TCB;
      if (TCCmp != order::equal)
        return TCCmp;
    }
  }

  return order::equal;
}

///\brief Compare two subtrees, saving the visited nodes onto two stacks
static std::tuple<order, EdgeList, EdgeList>
exploreAndCompare(const Link &Child1, const Link &Child2);

///\brief Recursively define an ordering between children of a node
static bool linkOrderLess(const Link &A, const Link &B) {
  const order EdgeOrder = cmpEdgeTags(A.second,
                                      B.second,
                                      /*IgnoreInheritance=*/false);
  if (EdgeOrder != order::equal)
    return EdgeOrder < 0;

  const order NodeOrder = cmpNodes(A.first, B.first);
  if (NodeOrder != order::equal)
    return NodeOrder < 0;

  revng_log(CmpLog,
            "No order between " << A.first->ID << " and " << A.first->ID
                                << ", must recur");
  // In case the two nodes are equivalent, explore the whole subtree
  // TODO: cache the result of this comparison?
  const auto [SubtreeOrder, _, __] = exploreAndCompare(A, B);
  revng_assert(SubtreeOrder != order::equal);
  return SubtreeOrder == order::less;
}

static std::tuple<order, EdgeList, EdgeList>
exploreAndCompare(const Link &Child1, const Link &Child2) {
  if (Child1.first->ID == Child2.first->ID)
    return { order::equal, { Child1 }, { Child2 } };

  EdgeList VisitStack1{ Child1 }, VisitStack2{ Child2 };
  EdgeList NextToVisit1, NextToVisit2;
  size_t CurIdx = 0;
  do {
    // Append the newly found nodes to the visit stack of each subtree
    size_t NextSize = NextToVisit1.size();
    revng_assert(NextSize == NextToVisit2.size());

    if (NextSize > 0) {
      size_t PrevSize = VisitStack1.size();
      VisitStack1.reserve(PrevSize + NextSize);
      VisitStack2.reserve(PrevSize + NextSize);
      VisitStack1.insert(VisitStack1.end(),
                         std::make_move_iterator(NextToVisit1.begin()),
                         std::make_move_iterator(NextToVisit1.end()));
      VisitStack2.insert(VisitStack2.end(),
                         std::make_move_iterator(NextToVisit2.begin()),
                         std::make_move_iterator(NextToVisit2.end()));
      NextToVisit1.clear();
      NextToVisit2.clear();
    }

    // Perform bfs on new nodes
    for (; CurIdx < VisitStack1.size(); CurIdx++) {
      const auto &[Node1, Edge1] = VisitStack1[CurIdx];
      const auto &[Node2, Edge2] = VisitStack2[CurIdx];
      revng_log(CmpLog, "Comparing " << Node1->ID << " with " << Node2->ID);

      if (Node1->ID == Node2->ID)
        continue;

      // TODO: handle pointer edges
      const order EdgeOrder = cmpEdgeTags(Edge1, Edge2);
      if (EdgeOrder != order::equal)
        return { EdgeOrder, {}, {} };

      const order NodeOrder = cmpNodes(Node1, Node2);
      if (NodeOrder != order::equal)
        return { NodeOrder, {}, {} };

      // Enqueue the successors of the current nodes
      revng_log(CmpLog, "Could not tell the difference, visiting successors");
      NextToVisit1.reserve(NextToVisit1.size() + Node1->Successors.size());
      NextToVisit2.reserve(NextToVisit2.size() + Node2->Successors.size());
      llvm::copy(Node1->Successors, std::back_inserter(NextToVisit1));
      llvm::copy(Node2->Successors, std::back_inserter(NextToVisit2));

      // Sort the newly enqueued nodes
      size_t NChildren = Node1->Successors.size();
      revng_assert(NChildren == Node2->Successors.size());

      std::sort(NextToVisit1.end() - NChildren,
                NextToVisit1.end(),
                linkOrderLess);
      std::sort(NextToVisit2.end() - NChildren,
                NextToVisit2.end(),
                linkOrderLess);
    }
  } while (NextToVisit1.size() > 0);

  return { order::equal, VisitStack1, VisitStack2 };
}

///\brief Check if two subtrees are equivalent, saving the visited nodes in the
/// order in which they were compared.
static inline std::tuple<bool, EdgeList, EdgeList>
areEquivSubtrees(Link &Child1, Link &Child2) {
  auto [Result, Visited1, Visited2] = exploreAndCompare(Child1, Child2);
  bool AreSubtreesEqual = Result == order::equal;

  return { AreSubtreesEqual, Visited1, Visited2 };
}

///\brief Visit the two subtrees of \a Child1 and \a Child2. If they are
/// equivalent, merge each node with the one it has been compared to.
///
///\return true if the two nodes were merged, and merged subtree
///\param TS the graph in which the comparison should be performed
///\param Child1 the root of the first subtree
///\param Child2 the root of the second subtree, will be collapsed if
///           equivalent to the subtree of \a Child1
static std::pair<bool, EdgeList>
mergeIfTopologicallyEq(LayoutTypeSystem &TS, Link &Child1, Link &Child2) {
  if (Child1 == Child2)
    return { true, {} };

  auto [AreEquiv, Subtree1, Subtree2] = areEquivSubtrees(Child1, Child2);
  if (AreEquiv) {
    revng_log(CmpLog, "Equivalent!");
    // Create a map between nodes to merge and the corresponding merge
    // destination, in order to:
    // 1. avoid duplicates in merging list
    // 2. check that a node is never merged into two separate nodes
    // 3. handle the case in which the merge destination has to be merged itself
    std::map</*to merge*/ LTSN *, /*to keep*/ LTSN *> MergeMap;
    for (const auto &[Link1, Link2] : llvm::zip(Subtree1, Subtree2)) {
      auto *NodeToKeep = Link1.first;
      auto *NodeToMerge = Link2.first;

      const auto &[_, Inserted] = MergeMap.insert({ NodeToMerge, NodeToKeep });
      if (not Inserted)
        revng_assert(MergeMap.at(NodeToMerge) = NodeToKeep);
    }

    // Redirect chains of nodes that have to be merged together
    llvm::SmallPtrSet<LTSN *, 8> Subtree1MergedNodes;
    for (auto &[NodeToMerge, NodeToKeep] : MergeMap) {
      if (NodeToKeep == NodeToMerge
          or Subtree1MergedNodes.contains(NodeToMerge))
        continue;

      auto MapEntry = MergeMap.find(NodeToKeep);
      llvm::SmallPtrSet<LTSN *, 8> MergeChain;

      // Find chains of nodes to merge
      while (MapEntry != MergeMap.end()) {
        Subtree1MergedNodes.insert(MapEntry->first);
        const auto &[_, Inserted] = MergeChain.insert(NodeToKeep);
        // Avoid loops
        if (not Inserted)
          break;

        NodeToKeep = MapEntry->second;

        // Go to next node of the chain
        MapEntry = MergeMap.find(NodeToKeep);
      }

      // Update the merge destination of all the nodes of the chain
      for (auto *N : MergeChain)
        MergeMap.at(N) = NodeToKeep;
    }

    // Execute merge
    for (auto &[NodeToMerge, NodeToKeep] : MergeMap) {
      if (NodeToKeep == NodeToMerge)
        continue;

      // TODO: light merge
      TS.mergeNodes({ NodeToKeep, NodeToMerge });
    }

    // Remove merged nodes from subtree1
    if (Subtree1MergedNodes.size() > 0) {
      for (auto It = Subtree1.begin(); It != Subtree1.end();) {
        if (Subtree1MergedNodes.contains(It->first))
          It = Subtree1.erase(It);
        else
          ++It;
      }
    }

    return { true, Subtree1 };
  }

  revng_log(CmpLog, "Different!");
  return { false, {} };
}

///\brief Remove conflicting edges and collapse single children after merging.
static bool
postProcessMerge(LayoutTypeSystem &TS, const EdgeList &MergedSubtree) {
  bool Modified = false;

  // Merging nodes together might have created conflicting edges, i.e.
  // instance-offset-0 edges that connect two nodes with an already
  // existing inheritance edges: remove them.
  for (auto &E : MergedSubtree) {
    // Materialize predecessors to avoid iterator invalidation
    llvm::SmallVector<LTSN *, 8> PredNodes;
    for (auto &PredLink : E.first->Predecessors)
      PredNodes.push_back(PredLink.first);

    // Remove conflicts from predecessors
    for (auto &Pred : PredNodes)
      Modified |= RemoveConflictingEdges::removeConflicts(TS, Pred);

    // Remove conflict from node
    Modified |= RemoveConflictingEdges::removeConflicts(TS, E.first);
  }

  // Merging nodes and removing conflicts might have created situations in
  // which a node has a single child: collapse it into its parent.
  LTSN *SubtreeRoot = MergedSubtree.begin()->first;
  for (auto &N : post_order(SubtreeRoot))
    Modified |= CollapseSingleChild::collapseSingle(TS, N);

  return Modified;
}

bool DeduplicateUnionFields::runOnTypeSystem(LayoutTypeSystem &TS) {
  bool TypeSystemChanged = false;
  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyConsistency() and TS.verifyDAG()
                 and TS.verifyInheritanceTree());

  if (Log.isEnabled())
    TS.dumpDotOnFile("before-deduplicate-union-fields.dot");

  for (LTSN *Root : llvm::nodes(&TS)) {
    revng_assert(Root != nullptr);
    if (not isRoot(Root))
      continue;

    // Visit all Union nodes in post-order
    for (LTSN *UnionNode : post_order(Root)) {
      if (UnionNode->InterferingInfo != AllChildrenAreInterfering)
        continue;
      revng_log(Log, "****** Union Node found: " << UnionNode->ID);

      llvm::SmallSetVector<Link, 8> ToCompare;
      llvm::SmallSetVector<Link, 8> Visited;

      for (Link Succ : UnionNode->Successors)
        ToCompare.insert(Succ);

      bool UnionNodeChanged = false;
      while (ToCompare.size() > 0) {
        Link CurLink = ToCompare.back();
        LTSN *CurChild = CurLink.first;
        ToCompare.pop_back();

        // Compare each pair of children of the union node
        bool Merged = false;
        for (Link VisitedLink : Visited) {
          LTSN *VisitedChild = VisitedLink.first;

          revng_log(Log, "Is " << CurChild->ID << " == " << VisitedChild->ID);
          revng_assert(VisitedChild != CurChild
                       or cmpEdgeTags(VisitedLink.second, CurLink.second) != 0);

          auto [IsMerged, MergedSubtree] = mergeIfTopologicallyEq(TS,
                                                                  VisitedLink,
                                                                  CurLink);

          if (IsMerged) {
            TypeSystemChanged = true;
            UnionNodeChanged = true;
            Merged = true;
            revng_log(Log, "Merged!");

            bool SubtreeChanged = postProcessMerge(TS, MergedSubtree);
            if (SubtreeChanged) {
              // If the subtree was modified, re-enqueue the node
              Visited.remove(VisitedLink);
              ToCompare.insert(VisitedLink);
            }

            // If the node was merged, stop comparing it with other children
            break;
          }
        }

        if (not Merged) {
          Visited.insert(CurLink);
          revng_log(Log, "Child " << CurChild->ID << " not merged");
        }
      }

      // Collapse the union node if we are left with only one member
      if (UnionNodeChanged) {
        CollapseSingleChild::collapseSingle(TS, UnionNode);
        RemoveConflictingEdges::removeConflicts(TS, UnionNode);
      }
    }
  }

  if (Log.isEnabled())
    TS.dumpDotOnFile("after-deduplicate-union-fields.dot");
  if (VerifyLog.isEnabled()) {
    revng_assert(TS.verifyConsistency());
    revng_assert(TS.verifyInheritanceDAG());
    revng_assert(TS.verifyInheritanceTree());
    revng_assert(TS.verifyConflicts());
  }

  return TypeSystemChanged;
}

} // end namespace dla
