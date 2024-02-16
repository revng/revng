//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>
#include <compare>
#include <cstdint>
#include <iterator>

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"

#include "DLAStep.h"

using LTSN = dla::LayoutTypeSystemNode;
using order = std::strong_ordering;
using Link = dla::LayoutTypeSystemNode::Link;
using EdgeList = std::vector<Link>;
using Tag = dla::TypeLinkTag;
using NonPointerFilterT = EdgeFilteredGraph<LTSN *, dla::isNotPointerEdge>;

using namespace llvm;

static Logger<> Log("dla-deduplicate-union-fields");
static Logger<> CmpLog("dla-duf-comparisons");

namespace dla {

/// Strong ordering for nodes: order by size, then by number of successors
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

  return order::equal;
}

/// Strong ordering for edges: order by kind, then by offset expression
static order cmpEdgeTags(const Tag *A, const Tag *B) {
  if (A == B)
    return order::equal;
  revng_assert(A != nullptr and B != nullptr);

  auto KindA = A->getKind();
  auto KindB = B->getKind();

  // If at least one is a pointer, only look at the kind
  if (KindA == TypeLinkTag::LK_Pointer or KindB == TypeLinkTag::LK_Pointer)
    return KindA <=> KindB;

  OffsetExpression OffA = A->getOffsetExpr();
  OffsetExpression OffB = B->getOffsetExpr();

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

/// Strong ordering for links: compare edge tags and destination node
static order cmpLinks(const Link &A, const Link &B) {
  const order EdgeOrder = cmpEdgeTags(A.second, B.second);
  if (EdgeOrder != order::equal)
    return EdgeOrder;

  // Pointer edges are equivalent only if they correspond to the same node
  if (isPointerEdge(A)) {
    revng_assert(isPointerEdge(B));
    return A.first->ID <=> B.first->ID;
  }

  const order NodeOrder = cmpNodes(A.first, B.first);
  if (NodeOrder != order::equal)
    return NodeOrder;

  return order::equal;
}

/// Compare two subtrees, saving the visited nodes onto two stacks
static std::tuple<order, EdgeList, EdgeList>
exploreAndCompare(const Link &Child1, const Link &Child2);

/// Recursively define an ordering between children of a node
static bool linkOrderLess(const Link &A, const Link &B) {
  if (A == B)
    return false;

  const order LinkOrder = cmpLinks(A, B);
  if (LinkOrder != order::equal)
    return LinkOrder < 0;

  revng_log(CmpLog,
            "No order between " << A.first->ID << " and " << B.first->ID
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
    for (; CurIdx < VisitStack1.size(); ++CurIdx) {
      const Link &L1 = VisitStack1[CurIdx];
      const Link &L2 = VisitStack2[CurIdx];
      const auto &[Node1, Edge1] = L1;
      const auto &[Node2, Edge2] = L2;
      revng_log(CmpLog, "Comparing " << Node1->ID << " with " << Node2->ID);

      if (Node1->ID == Node2->ID)
        continue;

      // Return if the links are different
      const order LinkOrder = cmpLinks(L1, L2);
      if (LinkOrder != order::equal)
        return { LinkOrder, VisitStack1, VisitStack2 };

      revng_log(CmpLog, "Could not tell the difference");

      if (not isPointerEdge(L1)) {
        // Enqueue the successors of the current nodes
        revng_assert(not isPointerEdge(L2));
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
    }
  } while (NextToVisit1.size() > 0);

  return { order::equal, VisitStack1, VisitStack2 };
}

/// Check if two subtrees are equivalent, saving the visited nodes in the
/// order in which they were compared.
static std::tuple<bool, EdgeList, EdgeList>
areEquivSubtrees(const Link &Child1, const Link &Child2) {
  auto [Result, Visited1, Visited2] = exploreAndCompare(Child1, Child2);
  bool AreSubtreesEqual = Result == order::equal;

  return { AreSubtreesEqual, Visited1, Visited2 };
}

/// Visit the two subtrees of \a Child1 and \a Child2. If they are
/// equivalent, merge each node with the one it has been compared to.
///
///\return true if the two nodes were merged, and merged subtree
///\param TS the graph in which the comparison should be performed
///\param Child1 the root of the first subtree
///\param Child2 the root of the second subtree, will be collapsed if
///           equivalent to the subtree of \a Child1
static std::tuple<bool, std::set<LTSN *>, std::set<LTSN *>>
mergeIfTopologicallyEq(LayoutTypeSystem &TS,
                       const Link &Child1,
                       const Link &Child2) {
  if (Child1.first == Child2.first) {
    revng_log(CmpLog, "Same Node!");
    return { false, {}, {} };
  }

  auto [AreEquiv, Subtree1, Subtree2] = areEquivSubtrees(Child1, Child2);
  if (not AreEquiv) {
    revng_log(CmpLog, "Different!");
    return { false, {}, {} };
  }

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
    revng_assert(Inserted or MergeMap.at(NodeToMerge) == NodeToKeep);
  }

  // Redirect chains of nodes that have to be merged together
  llvm::SmallPtrSet<LTSN *, 8> Subtree1MergedNodes;
  for (auto &[NodeToMerge, NodeToKeep] : MergeMap) {
    if (NodeToKeep == NodeToMerge or Subtree1MergedNodes.contains(NodeToMerge))
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
  std::set<LTSN *> ErasedNodes;
  for (auto &[NodeToMerge, NodeToKeep] : MergeMap) {
    if (NodeToKeep == NodeToMerge)
      continue;

    TS.mergeNodes({ NodeToKeep, NodeToMerge });
    ErasedNodes.insert(NodeToMerge);
  }

  // Build the set of preserved nodes
  std::set<LTSN *> Preserved;
  for (const auto &[Node, Tag] : Subtree1)
    if (not Subtree1MergedNodes.contains(Node))
      Preserved.insert(Node);

  // The root of Subtree1 should always be preserved
  revng_assert(Preserved.contains(Child1.first));

  return { true, Preserved, ErasedNodes };
}

static auto getSuccEdgesToChild(LTSN *Parent, LTSN *Child) {
  auto &Succ = Parent->Successors;
  using IDBasedKey = std::pair<uint64_t, const TypeLinkTag *>;
  return llvm::iterator_range(Succ.lower_bound(IDBasedKey{ Child->ID,
                                                           nullptr }),
                              Succ.upper_bound(IDBasedKey{ Child->ID + 1,
                                                           nullptr }));
}

bool DeduplicateFields::runOnTypeSystem(LayoutTypeSystem &TS) {
  bool TypeSystemChanged = false;
  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());

  llvm::SmallPtrSet<LTSN *, 16> VisitedNodes;

  for (LTSN *Root : llvm::nodes(&TS)) {
    revng_assert(Root != nullptr);
    if (not isRoot(Root))
      continue;

    llvm::SmallVector<LTSN *, 8> PostOrderFromRoot;
    for (LTSN *N : post_order_ext(NonPointerFilterT(Root), VisitedNodes)) {

      size_t NumInstanceEdges = 0;

      for ([[maybe_unused]] const auto &Edge :
           llvm::children_edges<NonPointerFilterT>(N))
        ++NumInstanceEdges;

      if (NumInstanceEdges < 2)
        continue;

      revng_log(Log, "****** Node with many fields found: " << N->ID);
      PostOrderFromRoot.push_back(N);
    }

    // Visit all nodes with fields in post-order. The post-order needs to be
    // cached because children can be merged during traversal, which would
    // invalidate iterators in llvm::post_order if we use it vanilla.
    for (LTSN *NodeWithFields : PostOrderFromRoot) {
      revng_log(Log,
                "****** Try to dedup children of NodeWithFields with ID: "
                  << NodeWithFields->ID);

      llvm::SmallSetVector<LTSN *, 8> FieldsToCompare;
      llvm::SmallSet<LTSN *, 8> OriginalFields;
      llvm::SmallSetVector<LTSN *, 8> AnalyzedNodesNotMerged;

      // We keep a separate list of successors since we might need to re-enqueue
      // some of them.
      revng_log(Log, "Children are:");
      LoggerIndent TmpIndent{ Log };
      for (const Link &L : NodeWithFields->Successors) {
        revng_log(Log, L.first->ID);
        FieldsToCompare.insert(L.first);
        OriginalFields.insert(L.first);
      }

      bool NodeWithFieldsChanged = false;
      while (FieldsToCompare.size() > 0) {
        LTSN *CurChild = FieldsToCompare.pop_back_val();

        LoggerIndent Indent{ Log };
        revng_log(Log, "Consider CurChild: " << CurChild->ID);

        // The CurChild can be connected to NodeWithFields with more than one
        // edge, so consider them all when comparing CurChild with the
        // AnalyzedNotMerged.
        bool FieldsMerged = false;
        auto CurChildEdges = getSuccEdgesToChild(NodeWithFields, CurChild);
        revng_log(Log,
                  "There are "
                    << std::distance(CurChildEdges.begin(), CurChildEdges.end())
                    << " edges from " << NodeWithFields->ID << " to "
                    << CurChild->ID);
        for (auto &CurLink : CurChildEdges) {

          LoggerIndent MoreIndent{ Log };
          revng_log(Log, "Edge: " << *CurLink.second);

          if (isPointerEdge(CurLink)) {
            revng_log(Log, "skip pointer edge");
            continue;
          }

          // We want to compare CurChild with all the other nodes that we have
          // looked at in previous iterations, and try to merge it with one of
          // them.
          for (LTSN *NotMergedNode : AnalyzedNodesNotMerged) {
            LoggerIndent MoreMoreIndent{ Log };
            revng_log(Log,
                      "Try to merge: " << CurLink.first->ID << " with "
                                       << NotMergedNode->ID);

            auto NotMergedEdges = getSuccEdgesToChild(NodeWithFields,
                                                      NotMergedNode);
            revng_log(Log,
                      "There are " << std::distance(NotMergedEdges.begin(),
                                                    NotMergedEdges.end())
                                   << " edges from " << NodeWithFields->ID
                                   << " to " << NotMergedNode->ID);

            bool AnalyzedNotMergedInvalidated = false;
            for (const Link &NotMergedLink : NotMergedEdges) {
              const auto &[NotMergedNode, NotMergedTag] = NotMergedLink;

              LoggerIndent MoreMoreIndent{ Log };
              revng_log(Log, "Edge to merge with: " << *NotMergedTag);

              if (isPointerEdge(NotMergedLink)) {
                revng_log(Log, "skip pointer edge");
              }

              auto [IsMerged,
                    Preserved,
                    Erased] = mergeIfTopologicallyEq(TS,
                                                     NotMergedLink,
                                                     CurLink);
              if (not IsMerged) {
                revng_log(Log, "Edge not merged!");
                continue;
              }
              revng_log(Log, "Edge merged!");

              // If we merged something, there should be at least one preserved
              // node and one erased node
              revng_assert(not Preserved.empty());
              revng_assert(not Erased.empty());

              // This should always be true, since whenever we merge we are at
              // least erasing CurChild, merging it with NotMergedNode.
              FieldsMerged |= Erased.contains(CurChild);
              revng_assert(FieldsMerged);

              TypeSystemChanged = true;
              NodeWithFieldsChanged = true;

              // Collapse new single children that could emerge while merging
              {
                // Copy the post_order into a SmallVector, since collapseSingle
                // might mutate the graph and screw up the po_iterator.
                for (auto &N : llvm::SmallVector<LTSN *>{
                       post_order(NonPointerFilterT(NotMergedNode)) })
                  CollapseSingleChild::collapseSingle(TS, N);

                // Notice that collapseSingle can actually remove more nodes.
                // In principle we should add them to Erased and remove them
                // from Preserved.
                // However, in the remainder of the code below, both Preserved
                // and Erased are only used to update FieldsToCompare and
                // AnalyzedNodesNotMerged, and to set boolean flags to control
                // iteration.
                // Hence, we can get away without updating Preserved and Erased,
                // since the following assertions hold.

                revng_assert(not Erased.contains(NotMergedNode));
                revng_assert(Preserved.contains(NotMergedNode));
              }

              revng_log(Log, "The merge has erased the following nodes:");
              for (auto &ErasedNode : Erased) {
                LoggerIndent MoreMoreMoreIndent{ Log };
                revng_log(Log, ErasedNode->ID);
                // The ErasedNode has been deleted while merging, so we never
                // want it to be processed again.
                bool Erased = FieldsToCompare.remove(ErasedNode);
                FieldsMerged |= Erased;
                Erased = AnalyzedNodesNotMerged.remove(ErasedNode);
                AnalyzedNotMergedInvalidated |= Erased;
              }

              revng_log(Log, "The merge has preserved the following nodes:");
              for (LTSN *PreservedNode : Preserved) {
                LoggerIndent MoreMoreMoreIndent{ Log };
                revng_log(Log, PreservedNode->ID);
                // The PreservedNode is preserved (not erased) by merge, but
                // the merge process might have changed it.
                // So, if it'a an original children of the NodeWithFields, we
                // need to re-process it, hence we add it to FieldsToCompare,
                // then set NodeWithFieldsChanged, and remove it from
                // AnalyzedNodesNotMerged.
                if (OriginalFields.contains(PreservedNode)) {
                  LoggerIndent MaxIndent{ Log };
                  revng_log(Log,
                            "Is an original field. Re-enqueue it for "
                            "comparison");
                  FieldsToCompare.insert(PreservedNode);
                  bool Erased = AnalyzedNodesNotMerged.remove(PreservedNode);
                  AnalyzedNotMergedInvalidated |= Erased;
                }
              }

              // This should always be true, since whenever we merge we are at
              // least preserving the NotMergedNode, which might have been
              // changed by the merge.
              revng_assert(AnalyzedNotMergedInvalidated);

              // We have merged the CurChild into NotMergedNode, we have to
              // brake out of all the loops looking at CurChild and at
              // AnalyzedNodesNotMerged, since both of these might have
              // changed.
              break;
            }

            if (AnalyzedNotMergedInvalidated) {
              // If we just merged CurChild into NotMergedNode we have
              // invalidated the AnalyzedNodesNotMerged iterators.
              // So we have to exit this loop and re-start iterating on
              // FieldsToCompare.
              break;
            }
          }

          // If the children of the NodeWithFields have been changed by the
          // merge, the CurChildEdges iterator ranges have been invalidated. So
          // we have to break out of this loop as well.
          if (FieldsMerged) {
            break;
          }
        }

        // If we haven't merged CurChild with anything we can mark it as
        // analyzed and not merged.
        if (not FieldsMerged) {
          AnalyzedNodesNotMerged.insert(CurChild);
          revng_log(Log, "CurChild " << CurChild->ID << " not merged");
        }
      }

      // Collapse the union node if we are left with only one member
      if (NodeWithFieldsChanged) {
        bool Changed = CollapseSingleChild::collapseSingle(TS, NodeWithFields);
        TypeSystemChanged |= Changed;
      }
    }
  }

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());

  return TypeSystemChanged;
}

} // end namespace dla
