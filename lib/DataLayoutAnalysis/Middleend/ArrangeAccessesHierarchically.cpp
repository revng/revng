//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include <algorithm>
#include <utility>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Support/Debug.h"

#include "DLAStep.h"
#include "FieldSizeComputation.h"

static Logger<> Log{ "sort-accesses-hierarchically" };

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
  if (isLeaf(Child) or hasPointerChild(Child) or hasPointerParent(Child)
      or hasStridedParent(Child) or hasAnotherParent(Child, Parent)) {
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

static PushThroughComparisonResult
makePushThroughComparisonResult(NeighborIterator ToBePushed,
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

  auto ThroughElemSize = ToBePushedThrough->first->Size;
  auto PushedFieldSize = getFieldSize(ToBePushed->first, ToBePushed->second);
  revng_assert(ThroughElemSize >= PushedFieldSize + Final.Offset);

  return PushThroughComparisonResult{ .ToPush = ToBePushed,
                                      .Through = ToBePushedThrough,
                                      .OEAfterPush = std::move(Final) };
}

// Compare the edges *AIt and *BIt to check if any of them can be pushed through
// the other. If so return proper info.
static std::optional<PushThroughComparisonResult>
canPushThrough(const NeighborIterator &AIt, const NeighborIterator &BIt) {
  // An edge can never be pushed through itself
  if (AIt == BIt)
    return std::nullopt;

  // If any edge is not an instance edge, the edges are not comparable.
  if (not isInstanceEdge(*AIt) or not isInstanceEdge(*BIt))
    return std::nullopt;

  // Here both edges are instance edges.

  const auto &[AChild, ATag] = *AIt;
  const auto &[BChild, BTag] = *BIt;

  if (isLeaf(AChild) and isLeaf(BChild))
    return std::nullopt;

  const auto &[AOffset, AStrides, ATripCounts] = ATag->getOffsetExpr();
  const auto &[BOffset, BStrides, BTripCounts] = BTag->getOffsetExpr();

  revng_assert(AStrides.size() == ATripCounts.size());
  revng_assert(BStrides.size() == BTripCounts.size());
  revng_assert(AStrides.empty() or AStrides.size() == 1);
  revng_assert(BStrides.empty() or BStrides.size() == 1);

  auto AFieldSize = getFieldSize(AChild, ATag);
  auto BFieldSize = getFieldSize(BChild, BTag);

  uint64_t AFieldEnd = AOffset + AFieldSize;
  uint64_t BFieldEnd = BOffset + BFieldSize;

  // If A and B occupy disjoint ranges of memory, none of them can be pushed
  // through the other, so they are not comparable.
  if (AFieldEnd <= (uint64_t) (BOffset))
    return std::nullopt;

  if (BFieldEnd <= (uint64_t) (AOffset))
    return std::nullopt;

  // Here we have the guarantee that A and B occupy partly overlapping ranges.
  // They are not necessarily comparable yet, because they could still be partly
  // overlapping and not included.
  // Detect the partly overlapping case and return unordered for them
  if (AOffset < BOffset and AFieldEnd < BFieldEnd)
    return std::nullopt;
  if (AOffset > BOffset and AFieldEnd > BFieldEnd)
    return std::nullopt;

  // The two fields start at the same point and have the same size. None of them
  // strictly includes the other, so they need to be structurally inspected to
  // be merged. There's a separate pass doing this later in DLA, so we're not
  // taking care of it now.
  if (AOffset == BOffset and AFieldEnd == BFieldEnd)
    return std::nullopt;

  // Here we have the guarantee that one of the following holds:
  // - A is included in or occupies the same range as B
  // - B is included in or occupies the same range as A
  // Detect what is the case.
  bool AIsOuter = AFieldSize > BFieldSize;

  auto OuterIt = AIsOuter ? AIt : BIt;
  const auto &[Outer, OuterTag] = *OuterIt;

  // If the larger node is a leaf we cannot push the other down, so we bail out.
  if (isLeaf(Outer))
    return std::nullopt;

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
  if (EndByteInElem > Outer->Size)
    return std::nullopt;

  return makePushThroughComparisonResult(InnerIt, OuterIt);
}

// Helper ordering for NeighborIterators. We need it here because we need to use
// such iterators and keys in associative containers.
// Notice that this might have undefined behaviour if dereferncing either LHS
// or RHS is undefined behaviour itself.
// The bottom line is that we should never insert invalid iterators into
// associative containers.
static std::weak_ordering
operator<=>(const NeighborIterator &LHS, const NeighborIterator &RHS) {
  const auto &[LHSSucc, LHSTag] = *LHS;
  const auto &[RHSSucc, RHSTag] = *RHS;
  if (auto Cmp = LHSSucc <=> RHSSucc; Cmp != 0)
    return Cmp;
  return LHSTag <=> RHSTag;
}

static llvm::SmallPtrSet<LayoutTypeSystemNode *, 8>
absorbVolatileChildren(LayoutTypeSystem &TS, LayoutTypeSystemNode *Parent) {

  llvm::SmallPtrSet<LayoutTypeSystemNode *, 8> Absorbed;

  if (isLeaf(Parent))
    return Absorbed;

  revng_assert(not hasPointerChild(Parent));

  for (auto VolatileChildren = getVolatileChildren(Parent);
       not VolatileChildren.empty();
       VolatileChildren = getVolatileChildren(Parent)) {

    struct InstanceEdge {
      OffsetExpression OE;
      LayoutTypeSystemNode *Target;

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
      TS.mergeNodes({ Parent, Volatile });
    }

    for (auto &[OffsetExpr, Target] : CompoundEdges.takeVector())
      TS.addInstanceLink(Parent, Target, std::move(OffsetExpr));
  }
  return Absorbed;
}

bool ArrangeAccessesHierarchically::runOnTypeSystem(LayoutTypeSystem &TS) {
  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());

  bool Changed = false;

  {
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

  // This dla::Step will heavily mutate the graph while iterating on it.
  // Despite thinking hard about it, we couldn't come up with a sane visit order
  // that could be pre-computed and guaranteed to be stable during the mutation,
  // or could be updated cheaply.
  // So we precompute some kind of global topological ordering, and use it as a
  // hard-coded ordering for analysys throughout the dla::Step.
  // Then we go fixed-point until we don't have anything left to recompute, but
  // we always follow this order because it's still better than iterating
  // randomly, or with some other ordering that is not stable.
  std::vector<LayoutTypeSystemNode *> Queue;
  std::set<const LayoutTypeSystemNode *> ToAnalyze;
  {

    std::set<const dla::LayoutTypeSystemNode *> Visited;
    for (LayoutTypeSystemNode *Root : llvm::nodes(&TS)) {
      revng_assert(Root != nullptr);
      if (not isRoot(Root))
        continue;

      for (LayoutTypeSystemNode *N :
           llvm::post_order_ext(NonPointerFilterT(Root), Visited)) {
        revng_assert(N->Size);
        Queue.push_back(N);
        ToAnalyze.insert(N);
      }
    }

    // Reverse the whole thing so it's more RPOT like, which is a topological
    // ordering of all the nodes in the graph.
    // The edges will change during the algorithm, but there's nothing we can do
    // about it.
    std::reverse(Queue.begin(), Queue.end());
  }

  while (not ToAnalyze.empty()) {

    for (LayoutTypeSystemNode *Parent : Queue) {

      // Erase Parent from ToAnalyze
      bool AlreadyAnalyzed = not ToAnalyze.erase(Parent);

      // If erasure failed, Parent wasn't in ToAnalyze, so it was already
      // analyzed in a previous iteration, and we can skip it.
      if (AlreadyAnalyzed)
        continue;

      revng_log(Log, "Analyzing parent: " << Parent->ID);

      // If we reach this point we have to analyze Parent, to see if some of its
      // Instance children can be pushed down some other of its Instance chilren

      // We want to look at all instance children edges of Parent, and compute a
      // partial ordering between them, with the relationship "A should be
      // pushed down B".

      // Before doing this we want to absorb potential other nodes that became
      // volatile when pushing them down.
      // For instance if we had A->B, A->C, B->C, and C was pushed down B, then
      // C becomes volatile inside B.
      //
      {
        auto Absorbed = absorbVolatileChildren(TS, Parent);
        for (auto *A : Absorbed)
          ToAnalyze.erase(A);
      }

      // We now define a custom graph.
      // Each node of this custom graph represents an instange edge of the
      // original LayoutTypeSystem, whose predecessor is Parent.
      // Edges of this custom graph represent a relationship between the custom
      // nodes (representing edges).
      // If a EdgeNode A has an edge towards an EdgeNode B it means that the
      // edge represented by A can be pushed through the edge represented by B.
      // Whenever the graph is build, we also attach to each edge of the custom
      // graph an OffsetExpression, that represents the computed final offset
      // after the push down.
      using EdgeNode = BidirectionalNode<NeighborIterator, OffsetExpression>;
      using EdgeInclusionGraph = GenericGraph<EdgeNode>;
      EdgeInclusionGraph EdgeInclusion;

      using GT = llvm::GraphTraits<LayoutTypeSystemNode *>;
      auto AIt = GT::child_edge_begin(Parent);
      auto ChildEnd = GT::child_edge_end(Parent);

      // Build an EdgeNode for each instance edge outgoing from Parent.
      std::map<NeighborIterator, EdgeNode *> ItEdgeNodes;
      for (; AIt != ChildEnd; ++AIt) {
        if (isInstanceEdge(*AIt))
          ItEdgeNodes[AIt] = EdgeInclusion.addNode(AIt);
      }

      // Compute the relationships that tell us if an edge can be pushed through
      // another.
      AIt = GT::child_edge_begin(Parent);
      for (; AIt != ChildEnd; ++AIt) {
        if (not isInstanceEdge(*AIt))
          continue;

        // If for some reason we find a non-fixed child it means that we
        // probably need to re-think this DLAStep in a fixed-point fashion
        // across the first part and the second push-down part.
        revng_assert(isFixedChild(Parent, AIt->first));

        for (auto BIt = std::next(AIt); BIt != ChildEnd; ++BIt) {

          auto MaybePushThrough = canPushThrough(AIt, BIt);
          if (not MaybePushThrough.has_value())
            continue;

          auto &[ToPush, Through, OEAfterPush] = MaybePushThrough.value();

          ItEdgeNodes.at(Through)->addPredecessor(ItEdgeNodes.at(ToPush),
                                                  OEAfterPush);

          revng_log(Log, "======================================");
          revng_log(Log, "Through: " << Through->first->ID);
          revng_log(Log,
                    "Through Off: " << Through->second->getOffsetExpr().Offset);
          if (not Through->second->getOffsetExpr().Strides.empty())
            revng_log(Log,
                      "Through Stride: "
                        << Through->second->getOffsetExpr().Strides.front());
          revng_log(Log, "-----");
          revng_log(Log, "ToPush: " << ToPush->first->ID);
          revng_log(Log,
                    "ToPush Off: " << ToPush->second->getOffsetExpr().Offset);
          if (not ToPush->second->getOffsetExpr().Strides.empty())
            revng_log(Log,
                      "ToPush Stride: "
                        << ToPush->second->getOffsetExpr().Strides.front());
          revng_log(Log, "-----");
          revng_log(Log, "Final Off: " << OEAfterPush.Offset);
          if (not OEAfterPush.Strides.empty())
            revng_log(Log, "Final Stride: " << OEAfterPush.Strides.front());
        }
      }

      llvm::SmallSet<NeighborIterator, 4> EdgesToErase;
      for (auto *EdgeNodeToPushThrough : EdgeInclusion.nodes()) {

        // If EdgeNodeToPushThrough has some successors, it means that there
        // are other edges across which it should be pushed through. At this
        // point we don't want to push the others down EdgeNodeToPushThrough,
        // because that operation could change the overall result on the graph.
        // So we back off. The other edges that could be pushed down through
        // EdgeNodeToPushThrough will be resolved at a later time.
        if (not EdgeNodeToPushThrough->successors().empty())
          continue;

        // If EdgeNodeToPushThrough has no predecessors, there is no other edge
        // to push through it, so we just skip it.
        if (EdgeNodeToPushThrough->predecessors().empty())
          continue;

        auto &ToPushThroughEdgeIt = EdgeNodeToPushThrough->data();
        auto *ToPushThrough = ToPushThroughEdgeIt->first;
        ToAnalyze.insert(ToPushThrough);

        for (auto &Edge : EdgeNodeToPushThrough->predecessor_edges()) {
          auto &[EdgeNodeToPushDown, FinalOE] = Edge;
          auto &PushedEdgeIt = EdgeNodeToPushDown->data();
          EdgesToErase.insert(PushedEdgeIt);
          auto *ToPushDown = PushedEdgeIt->first;

          revng_assert(not isLeaf(ToPushThrough));
          TS.addInstanceLink(ToPushThrough, ToPushDown, std::move(FinalOE));
        }
      }

      // Now finally clean up the edges that were pushed down.
      for (const auto &ToErase : EdgesToErase)
        TS.eraseEdge(Parent, ToErase);

      Changed |= not EdgesToErase.empty();
    }
  }

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());

  return Changed;
}

} // end namespace dla
