//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>
#include <compare>
#include <cstdint>
#include <iterator>

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "revng/DataLayoutAnalysis/DLATypeSystem.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#include "DLAStep.h"

using LTSN = dla::LayoutTypeSystemNode;
using order = std::strong_ordering;
using Link = dla::LayoutTypeSystemNode::Link;
using NodeVec = llvm::SmallVector<LTSN *>;
using EdgeVec = llvm::SmallVector<Link>;
using Tag = dla::TypeLinkTag;
using NonPointerFilterT = EdgeFilteredGraph<LTSN *, dla::isNotPointerEdge>;

using namespace llvm;

static Logger<> Log("dla-deduplicate-union-fields");
static Logger<> CmpLog("dla-duf-comparisons");

namespace dla {

/// Strong ordering for nodes: order by size, then by number of successors.
static std::weak_ordering weakOrderNodes(const LTSN *A, const LTSN *B) {
  if (A == B)
    return std::weak_ordering::equivalent;

  if (not A)
    return std::weak_ordering::less;

  if (not B)
    return std::weak_ordering::greater;

  const auto SizeCmp = A->Size <=> B->Size;
  if (SizeCmp != std::weak_ordering::equivalent) {
    revng_log(CmpLog, "Different sizes");
    return SizeCmp;
  }

  // If they have a different number of successors, order the one with less
  // successor first;
  size_t NChild1 = A->Successors.size();
  size_t NChild2 = B->Successors.size();
  const auto NChildCmp = NChild1 <=> NChild2;
  if (NChildCmp != std::weak_ordering::equivalent) {
    revng_log(CmpLog,
              "Different number of successors:  node "
                << A->ID << " has " << NChild1 << " successors, node " << B->ID
                << " has" << NChild2 << " successors");
    return NChildCmp;
  }

  // If they have the same number of successors, consider them by kind.
  size_t NInstanceChild1 = 0;
  size_t NInstanceChild2 = 0;
  size_t NPointerChild1 = 0;
  size_t NPointerChild2 = 0;
  for (const auto &[Child1, Child2] : llvm::zip(A->Successors, B->Successors)) {
    if (isInstanceEdge(Child1))
      ++NInstanceChild1;
    else
      ++NPointerChild1;

    if (isInstanceEdge(Child2))
      ++NInstanceChild2;
    else
      ++NPointerChild2;
  }

  if (auto Cmp = NInstanceChild1 <=> NInstanceChild2; Cmp != 0)
    return Cmp;

  if (auto Cmp = NPointerChild1 <=> NPointerChild2; Cmp != 0)
    return Cmp;

  // They are equivalent, but they are not equal, because they are actually 2
  // different nodes with a different ID.
  return std::weak_ordering::equivalent;
}

/// Strong ordering for nodes: order by size, then by number of successors, and
/// only at the end for ID.
static std::strong_ordering orderNodes(const LTSN *A, const LTSN *B) {
  std::weak_ordering WeakNodeCmp = weakOrderNodes(A, B);
  if (WeakNodeCmp != 0)
    return (WeakNodeCmp < 0) ? std::strong_ordering::less :
                               std::strong_ordering::greater;

  if (not A and not B)
    return std::strong_ordering::equal;

  return A->ID <=> B->ID;
}

/// Strong ordering for edges: order by kind, then by offset expression
static std::strong_ordering orderEdgeTags(const Tag *A, const Tag *B) {
  if (A == B)
    return std::strong_ordering::equal;

  revng_assert(A != nullptr and B != nullptr);

  auto KindA = A->getKind();
  auto KindB = B->getKind();

  // If they have different kind, only look at the kind
  const auto KindCmp = KindA <=> KindB;
  if (KindCmp != std::strong_ordering::equal)
    return KindCmp;

  OffsetExpression OffA = A->getOffsetExpr();
  OffsetExpression OffB = B->getOffsetExpr();

  // Smaller offsets go first
  const auto OffsetCmp = OffA.Offset <=> OffB.Offset;
  if (OffsetCmp != std::strong_ordering::equal)
    return OffsetCmp;

  // Smaller number of nested strides go first
  const auto StrideSizeCmp = OffA.Strides.size() <=> OffB.Strides.size();
  if (StrideSizeCmp != std::strong_ordering::equal)
    return StrideSizeCmp;

  // Lexical comparison of strides
  for (const auto &[StrA, StrB] : llvm::zip(OffA.Strides, OffB.Strides)) {
    const auto StrideCmp = StrA <=> StrB;
    if (StrideCmp != std::strong_ordering::equal)
      return StrideCmp;
  }

  // Smaller number of nested trip counts go first
  const auto TCSizeCmp = OffA.TripCounts.size() <=> OffB.TripCounts.size();
  if (TCSizeCmp != std::strong_ordering::equal)
    return TCSizeCmp;

  // Lexical comparison of nested trip counts
  for (const auto &[TCA, TCB] : llvm::zip(OffA.TripCounts, OffB.TripCounts)) {
    if (not TCA and not TCB)
      continue;
    if (TCA and not TCB)
      return std::strong_ordering::less;
    if (not TCA and TCB)
      return std::strong_ordering::greater;

    if (TCA and TCB) {
      const auto TCCmp = *TCA <=> *TCB;
      if (TCCmp != std::strong_ordering::equal)
        return TCCmp;
    }
  }

  return std::strong_ordering::equal;
}

/// Strong ordering for links: compare edge tags and destination node
static bool orderLinks(const Link &A, const Link &B) {
  if (std::strong_ordering EdgeOrder = orderEdgeTags(A.second, B.second);
      EdgeOrder != order::equal)
    return EdgeOrder < 0;

  return orderNodes(A.first, B.first) < 0;
}

/// Compare two subtrees, saving the visited nodes onto two stacks
static std::tuple<bool, NodeVec, NodeVec>
exploreAndCompare(const Link &Child1, const Link &Child2) {

  if (orderEdgeTags(Child1.second, Child2.second) != 0) {
    revng_log(CmpLog, "Different edges!");
    return { false, {}, {} };
  }

  NodeVec VisitStack1{ Child1.first }, VisitStack2{ Child2.first };

  for (size_t CurIdx = 0; CurIdx < VisitStack1.size(); ++CurIdx) {
    revng_assert(VisitStack1.size() == VisitStack2.size());

    const LTSN *Node1 = VisitStack1[CurIdx];
    const LTSN *Node2 = VisitStack2[CurIdx];

    revng_log(CmpLog, "Comparing " << Node1->ID << " with " << Node2->ID);

    // If they are the same there's no need to recur. From here downwards the 2
    // visits are guaranteed to always be the same.
    if (Node1 == Node2)
      continue;

    // If the nodes have different sizes, or different number of successors
    // we can already say they're not equivalent and bail out.
    if (auto WeakNodeCmp = weakOrderNodes(Node1, Node2); WeakNodeCmp != 0) {
      revng_log(CmpLog, "Not equivalent");
      return { false, {}, {} };
    }

    // Enqueue the successors of the current nodes
    EdgeVec NextToVisit1, NextToVisit2;
    size_t NChildren = Node1->Successors.size();
    revng_assert(NChildren == Node2->Successors.size());

    NextToVisit1.reserve(NChildren);
    NextToVisit2.reserve(NChildren);
    llvm::copy(Node1->Successors, std::back_inserter(NextToVisit1));
    llvm::copy(Node2->Successors, std::back_inserter(NextToVisit2));

    // Sort the newly enqueued instance children
    llvm::sort(NextToVisit1, orderLinks);
    llvm::sort(NextToVisit2, orderLinks);

    // Pointer edges could be traversed in principle, and make the
    // tree-comparison even deeper. But for now we prevent that, otherwise
    // the algorithm could end up running on very large parts of the graph,
    // and we'll also have to detect loops to avoid infinite iterations.
    // So for now we just compare pointers, that are always sorted at the
    // back, and bail out if we find a pair of pointer edges that point to
    // different nodes.
    // NOTE: This relies on the fact that pointer edges are sorted after all
    // other kinds of edges.
    size_t FirstPointerIndex = NChildren;
    size_t Index = 0;
    for (const auto &[Link1, Link2] : llvm::zip(NextToVisit1, NextToVisit2)) {
      if (isPointerEdge(Link1)) {
        revng_assert(isPointerEdge(Link2));
        if (Link1.first != Link2.first)
          return { false, {}, {} };
        if (FirstPointerIndex == NChildren)
          FirstPointerIndex = Index;
      }
      ++Index;
    }

    // Take note of the iterators pointing to the first pointer edge.
    auto NextToVisit1End = std::next(NextToVisit1.begin(), FirstPointerIndex);
    auto NextToVisit2End = std::next(NextToVisit2.begin(), FirstPointerIndex);

    // Insert in the visit stack all the non-pointer edges.
    // NOTE: if we ever want to make a deep traversal to compare pointers too,
    // this should be the only place to change. Just allow the traversal to
    // enqueue also all the pointer edges.
    auto NonPointerRange1 = llvm::make_range(NextToVisit1.begin(),
                                             NextToVisit1End);
    auto NextNodeRange1 = llvm::make_first_range(NonPointerRange1);
    VisitStack1.insert(VisitStack1.end(),
                       NextNodeRange1.begin(),
                       NextNodeRange1.end());

    auto NonPointerRange2 = llvm::make_range(NextToVisit2.begin(),
                                             NextToVisit2End);
    auto NextNodeRange2 = llvm::make_first_range(NonPointerRange2);
    VisitStack2.insert(VisitStack2.end(),
                       NextNodeRange2.begin(),
                       NextNodeRange2.end());
  }

  return { true, VisitStack1, VisitStack2 };
}

/// Check if two subtrees are equivalent, saving the visited nodes in the
/// order in which they were compared.
static std::tuple<bool, NodeVec, NodeVec> areEquivSubtrees(const Link &Child1,
                                                           const Link &Child2) {
  return exploreAndCompare(Child1, Child2);
}

/// Visit the two subtrees of \a ToKeep and \a ToMerge. If they are
/// equivalent, merge each node with the one it has been compared to.
///
///\return true if the two nodes were merged, and merged subtree
///\param TS the graph in which the comparison should be performed
///\param ToKeep the root of the first subtree
///\param ToMerge the root of the second subtree, will be collapsed if
///           equivalent to the subtree of \a ToKeep
static std::tuple<bool, std::set<LTSN *>, std::set<LTSN *>>
mergeIfTopologicallyEq(LayoutTypeSystem &TS,
                       const Link &ToKeep,
                       const Link &ToMerge) {

  auto [AreEquiv, Subtree1, Subtree2] = areEquivSubtrees(ToKeep, ToMerge);
  if (not AreEquiv) {
    revng_log(CmpLog, "Different!");
    return { false, {}, {} };
  }

  revng_log(CmpLog, "Equivalent!");

  // Compute equivalence classes of nodes that have to be merged together.
  llvm::EquivalenceClasses<LTSN *> MergeClasses;
  for (const auto &[N1, N2] : llvm::zip(Subtree1, Subtree2))
    MergeClasses.unionSets(N1, N2);

  LTSN *ChildToKeep = ToKeep.first;
  std::set<LTSN *> ErasedNodes;
  std::set<LTSN *> PreservedNodes;
  for (EquivalenceClasses<LTSN *>::iterator I = MergeClasses.begin(),
                                            E = MergeClasses.end();
       I != E;
       ++I) {

    if (!I->isLeader())
      continue; // Ignore non-leader sets.

    // Decide which node to keep for each equivalence class.
    // This typically depends only on the order of insertions of stuff in the
    // class, so it's deterministic. However, we want the NodeToKeep to always
    // be preserved, so we have to enforce it manually.
    LTSN *Keep = *MergeClasses.member_begin(I);
    if (MergeClasses.isEquivalent(ChildToKeep, Keep))
      Keep = ChildToKeep;

    for (LTSN *Merge : llvm::make_range(MergeClasses.member_begin(I),
                                        MergeClasses.member_end())) {
      if (Keep == Merge) {
        PreservedNodes.insert(Keep);
        continue;
      }
      TS.mergeNodes({ Keep, Merge });
      ErasedNodes.insert(Merge);
    }
  }

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());

  // The root of Subtree1 should always be preserved
  revng_assert(PreservedNodes.contains(ToKeep.first));

  return { true, PreservedNodes, ErasedNodes };
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

              // Check that the node that was meant to be preserved was actually
              // preserved, and not erased.
              // Notice that later collapseSingle can remove even more nodes.
              // In principle we should add them to Erased and remove them
              // from Preserved.
              // However, in the remainder of the code below, both Preserved
              // and Erased are only used to update FieldsToCompare and
              // AnalyzedNodesNotMerged, and to set boolean flags to control
              // iteration.
              // Hence, we can get away without updating Preserved and Erased.
              revng_assert(not Erased.contains(NotMergedNode));
              revng_assert(Preserved.contains(NotMergedNode));

              // This should always be true, since whenever we merge we are at
              // least erasing CurChild, merging it with NotMergedNode.
              FieldsMerged |= Erased.contains(CurChild);
              revng_assert(FieldsMerged);

              TypeSystemChanged = true;
              NodeWithFieldsChanged = true;

              // Collapse new single children that could emerge while merging.
              // Copy the post_order into a SmallVector, since collapseSingle
              // might mutate the graph and screw up the po_iterator.
              for (auto &N : llvm::SmallVector<LTSN *>{
                     post_order(NonPointerFilterT(NotMergedNode)) })
                CollapseSingleChild::collapseSingle(TS, N);

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
