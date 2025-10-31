//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <utility>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Support/Debug.h"

#include "DLAStep.h"
#include "FieldSizeComputation.h"

static Logger Log{ "sort-accesses-hierarchically" };

namespace dla {

// Returns true if N has an incoming instance strided edge
static bool hasStridedParent(const LayoutTypeSystemNode *N) {
  using InstanceGraph = EdgeFilteredGraph<const dla::LayoutTypeSystemNode *,
                                          dla::isNotPointerEdge>;
  using InverseInstance = llvm::Inverse<InstanceGraph>;
  for (const auto &Edge : llvm::children_edges<InverseInstance>(N))
    if (isInstanceEdge(Edge)
        and not Edge.second->getOffsetExpr().Strides.empty())
      return true;
  return false;
}

// Returns true if N has an incoming instance edge from a node different from
// Parent
static bool hasAnotherParent(const LayoutTypeSystemNode *N,
                             const LayoutTypeSystemNode *Parent) {
  using InstanceGraph = EdgeFilteredGraph<const dla::LayoutTypeSystemNode *,
                                          dla::isNotPointerEdge>;
  using InverseInstance = llvm::Inverse<InstanceGraph>;
  for (const auto *P : llvm::children<InverseInstance>(N))
    if (P != Parent)
      return true;

  return false;
}

// Returns true if N has a pointer successor (predecessor if OnInverse is true)
template<bool OnInverse>
static bool hasPointerSuccessor(const LayoutTypeSystemNode *N) {
  using ConstPointerGraph = EdgeFilteredGraph<const dla::LayoutTypeSystemNode *,
                                              dla::isPointerEdge>;
  using Directed = std::conditional_t<OnInverse,
                                      llvm::Inverse<ConstPointerGraph>,
                                      ConstPointerGraph>;

  for ([[maybe_unused]] const auto *_ : llvm::children<Directed>(N))
    return true;

  return false;
}

// Returns true if N has an incoming pointer edge
static bool hasPointerParent(const LayoutTypeSystemNode *N) {
  return hasPointerSuccessor<true>(N);
}

// Returns true if N has an outgoing pointer edge
static bool hasPointerChild(const LayoutTypeSystemNode *N) {
  return hasPointerSuccessor<false>(N);
}

// Returns true if Child is an instance child of Parent that should not be
// destroyed by this dla::Step.
static bool isFixedChild(const LayoutTypeSystemNode *Parent,
                         const LayoutTypeSystemNode *Child) {

  // If is'a a leaf node it's fixed because it represents an access.
  // If Child has an outgoing or incoming pointer edge, then it's fixed.
  // If there's a strided edge going to Child, then Child is fixed.
  // If Child has another Parent that is different from Parent, then it's
  // fixed
  if (Child->NonScalar or isLeaf(Child) or hasPointerChild(Child)
      or hasPointerParent(Child) or hasStridedParent(Child)
      or hasAnotherParent(Child, Parent)) {
    return true;
  }

  // TODO: what if the Child has many incoming edges, and one comes
  // from Parent and the others all come from stuff that's between
  // Parent and Child? (e.g. P -> C1; P -> C2; C1 -> C2;) In the
  // current implementation C2 is considered fixed, because it has
  // another parent different from P.

  // TODO: what if the Child has both a in incoming strided edge and
  // a non-strided edge that come from Parent???
  //(e.g. P -offset-> C; P -strided-> C;)
  // In the current implementation C is considered fixed, because it
  // has an incoming strided edge.

  return false;
}

using NonPointerFilterT = EdgeFilteredGraph<dla::LayoutTypeSystemNode *,
                                            dla::isNotPointerEdge>;

static llvm::SetVector<LayoutTypeSystemNode *>
getVolatileChildren(LayoutTypeSystemNode *Parent) {
  llvm::SetVector<LayoutTypeSystemNode *> Result;
  for (auto *Child : llvm::children<NonPointerFilterT>(Parent))
    if (not isFixedChild(Parent, Child))
      Result.insert(Child);
  return Result;
}

using NeighborIterator = LayoutTypeSystem::NeighborIterator;

// This struct represents if an edge in LayoutTypeSystem can be pushed down
// another edge in LayoutTypeSystem, along with the resulting OffsetExpression
// if the push down takes place.
struct PushThroughComparisonResult {
  NeighborIterator ToPush;
  NeighborIterator Through;
  OffsetExpression OEAfterPush;
};

static OffsetExpression
computeOffsetAfterPush(NeighborIterator ToBePushed,
                       NeighborIterator ToBePushedThrough) {
  OffsetExpression Final;

  revng_assert(isInstanceEdge(*ToBePushed));
  revng_assert(isInstanceEdge(*ToBePushedThrough));

  const auto &ToPushOE = ToBePushed->second->getOffsetExpr();
  const auto &ThroughOE = ToBePushedThrough->second->getOffsetExpr();

  revng_assert(ToPushOE.Strides.empty() or ToPushOE.Strides.size() == 1);
  revng_assert(ThroughOE.Strides.empty() or ThroughOE.Strides.size() == 1);

  // The edge to push through always has the larger offset. Just subtract the
  // offset of the other edge.
  revng_assert(ToPushOE.Offset >= ThroughOE.Offset);
  Final.Offset = ToPushOE.Offset - ThroughOE.Offset;

  // If a strided edge is being pushed down another edge it means that the
  // whole array represented by the edge to push is contained inside the
  // element represented by the edge we're pushing it through (or a single
  // element of the array represented by the edge we're pushing it through, if
  // the edge we're pushing through is strided).
  // If the edge being pushed down is not strided, strides and trip counts are
  // empty.
  // So in all cases we can preserve Strides and TripCounts.
  Final.Strides = ToPushOE.Strides;
  Final.TripCounts = ToPushOE.TripCounts;

  // Pushing through a strided edge, we need to compute the reminder of the
  // offset inside the element of the array represented by the edge we're
  // pushing through.
  if (not ThroughOE.Strides.empty()) {
    revng_assert(ThroughOE.Strides.size() == 1);
    Final.Offset %= ThroughOE.Strides.front();
  }

  uint64_t ThroughElemSize = ToBePushedThrough->first->Size;
  uint64_t PushedFieldSize = getFieldSize(ToBePushed->first,
                                          ToBePushed->second);
  revng_assert(ThroughElemSize >= PushedFieldSize + Final.Offset);

  return Final;
}

// Compare the edges *AIt and *BIt to check if any of them can be pushed through
// the other. If so return proper info.
static std::optional<PushThroughComparisonResult>
canPushThrough(const NeighborIterator &AIt, const NeighborIterator &BIt) {
  revng_log(Log, "canPushThrough");
  LoggerIndent Indent{ Log };

  // An edge can never be pushed through itself
  if (AIt == BIt) {
    revng_log(Log, "nullopt: same");
    return std::nullopt;
  }

  // If any edge is not an instance edge, the edges are not comparable.
  if (not isInstanceEdge(*AIt) or not isInstanceEdge(*BIt)) {
    revng_log(Log, "nullopt: not instance");
    return std::nullopt;
  }

  // Here both edges are instance edges.

  const auto &[AChild, ATag] = *AIt;
  const auto &[BChild, BTag] = *BIt;

  if (isLeaf(AChild) and isLeaf(BChild)) {
    revng_log(Log, "nullopt: child");
    return std::nullopt;
  }

  const auto &[AOffset, AStrides, ATripCounts] = ATag->getOffsetExpr();
  const auto &[BOffset, BStrides, BTripCounts] = BTag->getOffsetExpr();

  revng_assert(AStrides.size() == ATripCounts.size());
  revng_assert(BStrides.size() == BTripCounts.size());
  revng_assert(AStrides.empty() or AStrides.size() == 1);
  revng_assert(BStrides.empty() or BStrides.size() == 1);

  uint64_t AFieldSize = getFieldSize(AChild, ATag);
  uint64_t BFieldSize = getFieldSize(BChild, BTag);

  uint64_t AFieldEnd = AOffset + AFieldSize;
  uint64_t BFieldEnd = BOffset + BFieldSize;

  // If A and B occupy disjoint ranges of memory, none of them can be pushed
  // through the other, so they are not comparable.
  if (AFieldEnd <= BOffset) {
    revng_log(Log, "nullopt: A ends before B starts");
    return std::nullopt;
  }

  if (BFieldEnd <= AOffset) {
    revng_log(Log, "nullopt: B ends before A starts");
    return std::nullopt;
  }

  // Here we have the guarantee that A and B occupy partly overlapping ranges.
  // They are not necessarily comparable yet, because they could still be partly
  // overlapping and not included.
  // Detect the partly overlapping case and return unordered for them
  if (AOffset < BOffset and AFieldEnd < BFieldEnd) {
    revng_log(Log, "nullopt: partly overlapping, A starts first");
    return std::nullopt;
  }
  if (AOffset > BOffset and AFieldEnd > BFieldEnd) {
    revng_log(Log, "nullopt: partly overlapping, B starts first");
    return std::nullopt;
  }

  // The two fields start at the same point and have the same size. None of them
  // strictly includes the other, so they need to be structurally inspected to
  // be merged. There's a separate pass doing this later in DLA, so we're not
  // taking care of it now.
  if (AOffset == BOffset and AFieldEnd == BFieldEnd) {
    revng_log(Log, "nullopt: equal ranges, needs structural inspection");
    return std::nullopt;
  }

  // Here we have the guarantee that one of the following holds:
  // - A is included in or occupies the same range as B
  // - B is included in or occupies the same range as A
  // Detect what is the case.
  bool AIsOuter = AFieldSize > BFieldSize;

  auto OuterIt = AIsOuter ? AIt : BIt;
  const auto &[Outer, OuterTag] = *OuterIt;

  // If the larger node is a leaf we cannot push the other down, so we bail out.
  if (isLeaf(Outer)) {
    revng_log(Log, "nullopt: outer is a leaf");
    return std::nullopt;
  }

  auto InnerIt = AIsOuter ? BIt : AIt;
  const auto &[Inner, InnerTag] = *InnerIt;

  const auto &InnerOffset = InnerTag->getOffsetExpr().Offset;
  const auto &[OuterOffset, OuterStrides, _] = OuterTag->getOffsetExpr();

  auto InnerFieldSize = AIsOuter ? BFieldSize : AFieldSize;
  auto OuterElemSize = OuterStrides.empty() ? Outer->Size :
                                              OuterStrides.front();

  // The inner fields starts at an higher offset (or equal) than the outer
  revng_assert(InnerOffset >= OuterOffset);
  auto OffsetAfterPush = InnerOffset - OuterOffset;

  auto OffsetInElem = OffsetAfterPush % OuterElemSize;
  auto EndByteInElem = OffsetInElem + InnerFieldSize;
  if (EndByteInElem > Outer->Size) {
    revng_log(Log,
              "nullopt: EndByteInElem: " << EndByteInElem << " > "
                                         << " Outer->Size: " << Outer->Size);
    return std::nullopt;
  }

  return PushThroughComparisonResult{
    .ToPush = InnerIt,
    .Through = OuterIt,
    .OEAfterPush = computeOffsetAfterPush(InnerIt, OuterIt)
  };
}

// Helper ordering for NeighborIterators. We need it here because we need to use
// such iterators and keys in associative containers.
// Notice that this might have undefined behaviour if dereferencing either LHS
// or RHS is undefined behaviour itself.
// The bottom line is that we should never insert invalid iterators into
// associative containers.
static std::weak_ordering operator<=>(const NeighborIterator &LHS,
                                      const NeighborIterator &RHS) {
  const auto &[LHSSucc, LHSTag] = *LHS;
  const auto &[RHSSucc, RHSTag] = *RHS;
  if (auto Cmp = LHSSucc <=> RHSSucc; Cmp != 0)
    return Cmp;
  return LHSTag <=> RHSTag;
}

static llvm::SmallPtrSet<LayoutTypeSystemNode *, 8>
absorbVolatileChildren(LayoutTypeSystem &TS, LayoutTypeSystemNode *Parent) {
  revng_log(Log, "absorbVolatileChildren of: " << Parent->ID);
  LoggerIndent Indent{ Log };

  llvm::SmallPtrSet<LayoutTypeSystemNode *, 8> Absorbed;

  if (isLeaf(Parent))
    return Absorbed;

  revng_assert(not hasPointerChild(Parent));

  for (auto VolatileChildren = getVolatileChildren(Parent);
       not VolatileChildren.empty();
       VolatileChildren = getVolatileChildren(Parent)) {

    struct InstanceEdge {
      OffsetExpression OE;
      LayoutTypeSystemNode *Target = nullptr;

      // Ordering and comparison
      std::strong_ordering operator<=>(const InstanceEdge &) const = default;
      bool operator==(const InstanceEdge &) const = default;
    };

    llvm::SetVector<InstanceEdge,
                    llvm::SmallVector<InstanceEdge, 16>,
                    llvm::SmallSet<InstanceEdge, 16>>
      CompoundEdges;

    for (const auto &[Child, Tag] :
         llvm::children_edges<NonPointerFilterT>(Parent)) {
      // Don't mess up fixed children
      if (not VolatileChildren.contains(Child))
        continue;

      OffsetExpression OE = Tag->getOffsetExpr();

      // At this point we know that Child doesn't have incoming or
      // outgoing pointer edges. So all the edges involving Child are
      // instance edges.
      for (const auto &[GrandChild, ChildTag] :
           llvm::children_edges<NonPointerFilterT>(Child)) {
        revng_assert(not VolatileChildren.contains(GrandChild));
        auto New = InstanceEdge{
          OffsetExpression::append(OE, ChildTag->getOffsetExpr()), GrandChild
        };
        CompoundEdges.insert(New);
      }
    }

    // Remove all volatile nodes
    for (LayoutTypeSystemNode *Volatile : VolatileChildren) {
      Absorbed.insert(Volatile);
      TS.dropOutgoingEdges(Volatile);
      revng_log(Log, "Merge volatile child: " << Volatile->ID);
      TS.mergeNodes({ Parent, Volatile });
    }

    for (auto &[OffsetExpr, Target] : CompoundEdges.takeVector())
      TS.addInstanceLink(Parent, Target, std::move(OffsetExpr));
  }
  return Absorbed;
}

static void absorbVolatileChildren(LayoutTypeSystem &TS) {
  std::set<const dla::LayoutTypeSystemNode *> Visited;
  for (LayoutTypeSystemNode *Root : llvm::nodes(&TS)) {

    if (Visited.contains(Root))
      continue;

    if (not isRoot(Root))
      continue;

    for (LayoutTypeSystemNode *Node :
         llvm::post_order_ext(NonPointerFilterT(Root), Visited))
      absorbVolatileChildren(TS, Node);
  }
}

bool ArrangeAccessesHierarchically::runOnTypeSystem(LayoutTypeSystem &TS) {
  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());

  bool Changed = false;

  absorbVolatileChildren(TS);

  // This dla::Step will heavily mutate the graph while iterating on it.
  // Despite thinking hard about it, we couldn't come up with a sane visit order
  // that could be pre-computed and guaranteed to be stable during the mutation,
  // or could be updated cheaply.
  // So we precompute some kind of global topological ordering, and use it as a
  // hard-coded ordering for analysis throughout the dla::Step.
  // Then we go fixed-point until we don't have anything left to recompute, but
  // we always follow this order because it's still better than iterating
  // randomly, or with some other ordering that is not stable.

  // Add a fake root to be able to compute a RPOT from that. What's important
  // here is not that it's strictly a RPOT, but that it is a topological
  // traversal.
  auto *FakeRoot = TS.createArtificialLayoutType();
  for (LayoutTypeSystemNode *Root : llvm::nodes(&TS)) {
    revng_assert(Root != nullptr);
    if (not isRoot(Root))
      continue;

    TS.addInstanceLink(FakeRoot, Root, OffsetExpression{});
  }

  // Compute the RPOT
  auto Queue = llvm::ReversePostOrderTraversal(NonPointerFilterT(FakeRoot));

  // Compute the node to analyze.
  std::set<const LayoutTypeSystemNode *> ToAnalyze;
  for (const auto *Node : Queue)
    ToAnalyze.insert(Node);

  // The fake root is never to analyze and can already be removed.
  ToAnalyze.erase(FakeRoot);
  TS.removeNode(FakeRoot);

  while (not ToAnalyze.empty()) {

    for (LayoutTypeSystemNode *Parent : Queue) {

      // Erase Parent from ToAnalyze
      bool AlreadyAnalyzed = not ToAnalyze.erase(Parent);

      // If erasure failed, Parent wasn't in ToAnalyze, so it was already
      // analyzed in a previous iteration, and we can skip it.
      if (AlreadyAnalyzed)
        continue;

      revng_log(Log, "Analyzing parent: " << Parent->ID);
      revng_assert(Parent != FakeRoot);
      LoggerIndent Indent{ Log };

      // If we reach this point we have to analyze Parent, to see if some of its
      // Instance children can be pushed down some other of its Instance
      // children

      // We want to look at all instance children edges of Parent, and compute a
      // partial ordering between them, with the relationship "A should be
      // pushed down B".

      // Before doing this we want to absorb potential other nodes that became
      // volatile when pushing them down.
      // For instance if we had A->B, A->C, B->C, and C was pushed down B, then
      // C becomes volatile inside B.
      {
        auto Absorbed = absorbVolatileChildren(TS, Parent);
        for (auto *A : Absorbed)
          ToAnalyze.erase(A);
      }

      // Represents an edge iterator pointing to an instance edge, along with an
      // OffsetExpression that is not the one currently attached to the instance
      // edge, but that it is used to move the edge around.
      struct EdgeWithOffset {
        NeighborIterator Edge;
        OffsetExpression FinalOE;
      };

      revng_log(Log, "Initialize ChildrenHierarchy");
      // A map whose key is an edge iterator representing an instance edge, and
      // the mapped type is a vector of other edges that can be pushed through
      // the key, along the updated offsets that they will have after pushing
      // them through it.
      // In particular, we only keep in this map the neighbor iterator that are
      // the roots of the hierarchy of instance edges, i.e. the instance edges
      // that cannot be pushed through any other edge.
      std::map<NeighborIterator, llvm::SmallVector<EdgeWithOffset>>
        ChildrenHierarchy;

      using GT = llvm::GraphTraits<LayoutTypeSystemNode *>;

      // Initialize the ChildrenHierarchy.
      // At the beginning, all children edges are roots, and none of them has
      // other edges that can be pushed through them.
      for (auto AIt = GT::child_edge_begin(Parent),
                ChildEnd = GT::child_edge_end(Parent);
           AIt != ChildEnd;
           ++AIt) {
        if (isInstanceEdge(*AIt))
          ChildrenHierarchy[AIt];
      }

      revng_log(Log, "Compare children edges");
      // Now compare each root only with other roots
      auto ARootIt = ChildrenHierarchy.begin();
      auto ARootNext = ARootIt;
      auto HierarchyEnd = ChildrenHierarchy.end();
      for (; ARootIt != HierarchyEnd; ARootIt = ARootNext) {
        LoggerIndent ChildIndent{ Log };

        auto &[AEdgeIt, PushedInsideA] = *ARootIt;

        revng_log(Log,
                  "comparing AEdge: " << AEdgeIt->first->ID
                                      << " label: " << AEdgeIt->second);

        // A vector of instance edges that are contained in AEdgeIt and that
        // should be pushed through it, along with the new offsets they will
        // have after the push through.
        llvm::SmallVector<EdgeWithOffset> ContainedInA;

        // A vector of instance edges that contain AEdgeIt. AEdgeIt will have to
        // be pushed through all of them. The FinalOE in EdgeWithOffset here
        // represents the new offset that AEdgeIt will have after being pushed
        // through them.
        llvm::SmallVector<EdgeWithOffset> ContainsA;

        for (auto BRootIt = ARootNext; BRootIt != HierarchyEnd; ++BRootIt) {
          LoggerIndent OtherChildIndent{ Log };

          auto &[BEdgeIt, ContainedInB] = *BRootIt;
          revng_log(Log,
                    "with BEdge: " << BEdgeIt->first->ID
                                   << " label: " << BEdgeIt->second);

          auto MaybePushThrough = canPushThrough(AEdgeIt, BEdgeIt);
          if (not MaybePushThrough.has_value())
            continue;

          auto &[ToPush, Through, OEAfterPush] = MaybePushThrough.value();
          revng_assert(ToPush == AEdgeIt or ToPush == BEdgeIt);
          revng_assert(Through == AEdgeIt or Through == BEdgeIt);
          revng_assert(Through != ToPush);
          if (ToPush == AEdgeIt) {
            revng_assert(Through == BEdgeIt);
            // AEdgeIt can be pushed inside BEdgeIt
            auto BWithNewAOffset = EdgeWithOffset{ BEdgeIt,
                                                   std::move(OEAfterPush) };
            ContainsA.push_back(std::move(BWithNewAOffset));
          } else {
            revng_assert(ToPush == BEdgeIt and Through == AEdgeIt);
            // BEdgeIt can be pushed inside AEdgeIt
            auto BWithNewBOffset = EdgeWithOffset{ BEdgeIt,
                                                   std::move(OEAfterPush) };
            ContainedInA.push_back(std::move(BWithNewBOffset));
          }
        }

        // Now, push all the roots contained in A into A, and remove them from
        // ChildrenHierarchy, since they're not roots anymore.
        PushedInsideA.reserve(PushedInsideA.size() + ContainedInA.size());
        PushedInsideA.append(ContainedInA);
        for (const auto &[PushedInA, Unused] : ContainedInA)
          ChildrenHierarchy.erase(PushedInA);

        // Then, for all the other edges that contain A, A can be pushed through
        // them, along with all the nodes that have already been pushed through
        // A itself (that transitively can be pushed through the others too).
        // After we've pushed A through all edges in ContainsA, we can remove A
        // itelf from ChildrenHierarchy, since it's not a root anymore.
        ARootNext = std::next(ARootIt);
        if (not ContainsA.empty()) {
          for (const auto &[LargerThanA, FinalOE] : ContainsA) {

            // If A needs to be pushed through LargerThanA, and all things that
            // were previously pushed pushed through A can now be pushed through
            // LargerThanA.
            auto &PushedThroughLargerThanA = ChildrenHierarchy.at(LargerThanA);
            PushedThroughLargerThanA.reserve(PushedThroughLargerThanA.size()
                                             + PushedInsideA.size() + 1);
            for (auto &[EdgeToPush, OEAfterPush] : PushedInsideA) {
              // We have to recompute the offset of EdgeToPush here, after being
              // pushed through LargerThanA, because this value was never
              // computed before, given that these two edges have never been
              // compared before.
              PushedThroughLargerThanA.push_back(EdgeWithOffset{
                EdgeToPush, computeOffsetAfterPush(EdgeToPush, LargerThanA) });
            }
            // Then also AEdgeIt can be pushed through LargerThanA. The final
            // offset of AEdgeIt after being pushed through has already been
            // computed in advance, so we just use that.
            PushedThroughLargerThanA.push_back({ AEdgeIt, std::move(FinalOE) });
          }
          // Now AEdgeIt is not a root anymore, so we have to erase it and
          // update ARootNext.
          ARootNext = ChildrenHierarchy.erase(ARootIt);
        }
      }

      llvm::SmallSet<NeighborIterator, 4> EdgesToErase;
      {
        revng_log(Log, "Collect EdgesToErase");
        LoggerIndent IndentCollect{ Log };
        for (auto &[EdgeToPushThrough, EdgesToPush] : ChildrenHierarchy) {

          if (EdgesToPush.empty())
            continue;

          auto *ToPushThrough = EdgeToPushThrough->first;
          ToAnalyze.insert(ToPushThrough);

          for (auto &[PushedEdgeIt, FinalOE] : EdgesToPush) {
            EdgesToErase.insert(PushedEdgeIt);
            auto *ToPushDown = PushedEdgeIt->first;

            revng_assert(not isLeaf(ToPushThrough));
            TS.addInstanceLink(ToPushThrough, ToPushDown, std::move(FinalOE));
          }
        }
      }

      // Now finally clean up the edges that were pushed down.
      {
        revng_log(Log, "Erase edges");
        LoggerIndent IndentErase{ Log };
        for (const auto &ToErase : EdgesToErase) {
          revng_log(Log,
                    "erase: " << Parent->ID << " -> " << ToErase->first->ID
                              << " label: " << ToErase->second);
          TS.eraseEdge(Parent, ToErase);
        }
      }

      Changed |= not EdgesToErase.empty();
    }
  }

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());

  return Changed;
}

} // end namespace dla
