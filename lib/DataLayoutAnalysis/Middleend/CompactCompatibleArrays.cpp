//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include <functional>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"

#include "DLAStep.h"

namespace dla {

using NeighborIterator = LayoutTypeSystem::NeighborIterator;

// Helper ordering for NeighborIterators. We need it here because we need to use
// such iterators and keys in associative containers, and we want neighbors with
// lower offset to come first.
// Notice that this might have undefined behavior if dereferncing either LHS
// or RHS is undefined behavior itself.
// The bottom line is that we should never insert invalid iterators into
// associative containers.
static std::weak_ordering
operator<=>(const NeighborIterator &LHS, const NeighborIterator &RHS) {
  const auto &[LHSSucc, LHSTag] = *LHS;
  const auto &[RHSSucc, RHSTag] = *RHS;
  if (auto Cmp = LHSTag <=> RHSTag; Cmp != 0)
    return Cmp;
  return LHSSucc <=> RHSSucc;
}

struct InstanceEdge {
  OffsetExpression OE;
  LayoutTypeSystemNode *Target;

  InstanceEdge() = default;

  template<typename OffsetExpressionT>
  requires std::is_same_v<std::remove_cvref_t<OffsetExpressionT>,
                          OffsetExpression>
  InstanceEdge(OffsetExpressionT &&O, LayoutTypeSystemNode *N) :
    OE(std::forward<OffsetExpressionT>(O)), Target(N) {}

  template<typename OffsetExpressionT>
  requires std::is_same_v<std::remove_cvref_t<OffsetExpressionT>,
                          OffsetExpression>
  InstanceEdge(OffsetExpressionT &&O) :
    InstanceEdge(std::forward<OffsetExpressionT>(O), nullptr) {}

  InstanceEdge(const LayoutTypeSystemNode::Link &L) :
    InstanceEdge(L.second->getOffsetExpr(), L.first) {}

  InstanceEdge(const InstanceEdge &) = default;
  InstanceEdge &operator=(const InstanceEdge &) = default;

  InstanceEdge(InstanceEdge &&) = default;
  InstanceEdge &operator=(InstanceEdge &&) = default;

  // Ordering and comparison
  std::strong_ordering operator<=>(const InstanceEdge &) const = default;
  bool operator==(const InstanceEdge &) const = default;
};

struct CompatibleArrayGroup {
  std::set<InstanceEdge> Neighbors;
  uint64_t NElems;
  uint64_t BaseOffset;
};

bool CompactCompatibleArrays::runOnTypeSystem(LayoutTypeSystem &TS) {
  bool Changed = false;

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());

  std::set<LayoutTypeSystemNode *> Visited;
  for (LayoutTypeSystemNode *Root : llvm::nodes(&TS)) {
    if (not isRoot(Root))
      continue;

    using NonPointerFilter = EdgeFilteredGraph<LayoutTypeSystemNode *,
                                               isNotPointerEdge>;

    std::vector<CompatibleArrayGroup> ArrayGroups;

    for (LayoutTypeSystemNode *Parent :
         llvm::post_order_ext(NonPointerFilter(Root), Visited)) {

      // Skip leaf nodes (pointer nodes and access nodes).
      if (isLeaf(Parent))
        continue;

      using GT = llvm::GraphTraits<LayoutTypeSystemNode *>;
      auto ChildEdgeIt = GT::child_edge_begin(Parent);
      auto ChildEdgeNext = ChildEdgeIt;
      auto ChildEdgeEnd = GT::child_edge_end(Parent);
      for (; ChildEdgeIt != ChildEdgeEnd; ChildEdgeIt = ChildEdgeNext) {

        ChildEdgeNext = std::next(ChildEdgeIt);

        // Ignore edges that are not strided.
        const auto &ArrayEdge = *ChildEdgeIt;
        if (not isStridedInstance(ArrayEdge))
          continue;

        // Here we're starting from Array, which is an array that has not been
        // grouped with other yet.

        // Now we want to look around Array, and see if there are other
        // conflicting siblings that may need to be compacted with Array itself,
        // possibly changing the trip count or the actual start of the array.

        // Let's compute the running variables that we'll use to track where the
        // compacted array starts and ends.
        // These will be updated in flight until we finish the group and decide
        // all the siblings that need to be compacted with Array.

        const auto &[InitialChild, OE] = ArrayEdge;
        const auto &[InitialOffset,
                     InitialStrides,
                     InitialTripCounts] = OE->getOffsetExpr();

        // The start offset of the final compacted array.
        // Initially it's just InitialChild Offset. This will never increase,
        // but it could decrease if we find strong evidence that the final
        // compacted array actually starts earlier than the point we're starting
        // from.
        uint64_t ArrayStartOffset = InitialOffset;

        // The stride of the array. This will actually never change, because
        // we're only going to look with others array to compact with this if
        // they have a matching Stride.
        uint64_t Stride = InitialStrides.front();

        // The trip count of the array.
        // This will evolve to track the trip count of the compacted array.
        // It will only increase, never decrease.
        uint64_t TripCount = InitialTripCounts.front().value_or(1ULL);

        // The end offset of the final compacted array.
        // Initially it's just the last byte that is guaranteed to be accessed
        // by InitialChild. This will never decrease, but it could become
        // bigger, if we find strong evidence in other siblngs that the
        // compacted array should actually be larger.
        uint64_t ArrayEndOffset = InitialOffset + (Stride * (TripCount - 1ULL))
                                  + InitialChild->Size;

        // The initial slack inside an array element.
        // It represents the number of empty trailing bytes at the end of an
        // array element.
        revng_assert(Stride >= InitialChild->Size);
        uint64_t AvailableSlack = Stride - InitialChild->Size;

        // Ok, now we start looking at other array siblings of Array.
        // If we find an ArraySibling that strongly overlaps with the array
        // we're tracking we compact them and update our running variables
        // (ArrayStartOffset, ArrayEndOffset, AvailableSlack).
        llvm::SetVector<NeighborIterator,
                        llvm::SmallVector<NeighborIterator, 8>,
                        llvm::SmallSet<NeighborIterator, 8>>
          CompactedWithCurrent;

        auto SiblingEdgeIt = GT::child_edge_begin(Parent);
        auto SiblingEdgeNext = SiblingEdgeIt;
        auto SiblingEdgeEnd = GT::child_edge_end(Parent);
        for (; SiblingEdgeIt != SiblingEdgeEnd;
             SiblingEdgeIt = SiblingEdgeNext) {

          SiblingEdgeNext = std::next(SiblingEdgeIt);
          // Ignore ChildEdgeIt, because it's the one we've started from. We're
          // trying to compact some other array on to it.
          if (SiblingEdgeIt == ChildEdgeIt)
            continue;

          // Ignore non strided edges for now. We want to compact arrays first,
          // and then consider non-strided instance edges only later, taking
          // into account only the slack left.
          const auto &ArraySiblingEdge = *SiblingEdgeIt;
          if (not isStridedInstance(ArraySiblingEdge))
            continue;

          const auto &[Sibling, SiblingOE] = ArraySiblingEdge;
          const auto &[SiblingOffset,
                       SiblingStrides,
                       SiblingTripCounts] = SiblingOE->getOffsetExpr();

          // If Sibling has a mismatching stride we don't compact it.
          revng_assert(SiblingStrides.size() == 1);
          if (SiblingStrides.front() != Stride)
            continue;

          // If the Sibling starts later than ArrayEndOffset, we don't compact
          // them.
          if (SiblingOffset >= ArrayEndOffset)
            continue;

          uint64_t SiblingTripCount = SiblingTripCounts.front().value_or(1ULL);
          uint64_t SiblingEndOffset = SiblingOffset
                                      + (Stride * (SiblingTripCount - 1ULL))
                                      + Sibling->Size;

          // If the Sibling ends before ArrayStartOffset, we don't compact
          // them.
          if (SiblingEndOffset <= ArrayStartOffset)
            continue;

          // Here we have a strong evidence that ArraySibling overlaps the range
          // of the compacted array we're tracking for at least one byte.
          // At this point we want to find out if we can compact them.

          // To make the following computation more uniform (so we need to
          // handle less corner cases) we want to always assume that
          // SiblingOffset >= ArrayStartOffset.
          if (ArrayStartOffset > SiblingOffset) {
            // We want to move the tracked range to the left so we end up with
            // ArrayStartOffset <= SiblingOffset.
            // This shift to the left will be done in 2 stages:
            // - first we align the elements of ArraySibling with the array
            //   range we're tracking, exploiting the AvailableSlack
            // - then we add a number of entrire elemenents to the left
            uint64_t OffsetDifference = ArrayStartOffset - SiblingOffset;

            // If we don't have enough slack we just don't compact ArraySibling
            // with this group of edges.
            uint64_t RequiredSlack = OffsetDifference % Stride;
            if (RequiredSlack > AvailableSlack)
              continue;

            // If we reach this point, we have the guarantee that we will not
            // bail out, and that ArraySibling will be compacted, so we can
            // change the AvailableSlack and ArrayStartOffset

            AvailableSlack -= RequiredSlack;
            ArrayStartOffset -= RequiredSlack;
            OffsetDifference -= RequiredSlack;

            // If there is still some OffsetDifference after adjusting the
            // slack, we have to add a whole bunch of new elements before
            // ArrayStartOffset.
            if (OffsetDifference) {
              // Count how many elements we have to add before the array we're
              // tracking.
              uint64_t NumPrecedingElements = OffsetDifference / Stride;

              // Effectively change the start of the range we're tracking as if
              // we added the proper amount of elements before the beginning. We
              // might need to subtract Stride once more (see below) if the
              // division above had non-zero remainder.
              ArrayStartOffset -= Stride * NumPrecedingElements;
            }

            revng_assert(ArrayStartOffset == SiblingOffset);
          }

          // When we reach this point, the range of the array we're tracking
          // starts earlier than the ArraySibling.

          uint64_t OffsetDifference = SiblingOffset - ArrayStartOffset;
          uint64_t OffsetOfSiblingInArray = OffsetDifference % Stride;

          // We need to realign the element of the array moving it a little bit
          // to the left, consuming slack. RequiredSlack tells us how much of
          // the AvailableSlack we would consume doing that.
          uint64_t RequiredSlack = 0ULL;
          if (OffsetOfSiblingInArray)
            RequiredSlack = Stride - OffsetOfSiblingInArray;

          // If we need slack but we don't have it, the Sibling will not be
          // compacted with the array we're tracking.
          if (RequiredSlack > AvailableSlack)
            continue;

          // Here we know that ArraySibling can be compacted with the current
          // array we're tracking.
          CompactedWithCurrent.insert(SiblingEdgeIt);

          // If the size of the range we're tracking is going to increase, we
          // have to restart looking at array siblings because, given that the
          // range increases, there could be new siblings that have
          // overlapping bytes, and we have to take them into consideration
          // too.
          // This may happen in 2 scenarios
          if (RequiredSlack or SiblingEndOffset > ArrayEndOffset)
            SiblingEdgeNext = GT::child_edge_begin(Parent);

          // Move the start of the array back, if we need some slack
          ArrayStartOffset -= RequiredSlack;

          // Update the available slack.
          // It may be smaller than AvailableSlack - RequiredSlack if
          // SiblingSize is large, because the compacted sibling will consume
          // more of the bytes in the Stride, effectively leaving less stride.
          AvailableSlack = std::min(AvailableSlack - RequiredSlack,
                                    Stride - Sibling->Size);

          // Update the ArrayEndOffset, given that the ArraySibling could end
          // at a higher offset.
          ArrayEndOffset = std::max(ArrayEndOffset, SiblingEndOffset);

          // Update the TripCount.
          // The new TripCount is computed by counting how many times the
          // Stride fits into the new [ArrayStartOffset, ArrayEndOffset)
          // range. If the range is not fully divisible by Stride, count an
          // extra element to allow space for the trailing stuff.
          // There's no need to check the remainder to see if it's not fully
          // divisible. It's enough to check the AvailableSlack.
          TripCount = (ArrayEndOffset - ArrayStartOffset) / Stride;
          if (AvailableSlack)
            ++TripCount;
        }

        // TODO: we should think if it's beneficial to incorporate single
        // elements, and how to treat arrays with unknown trip counts

        // If we have something to compact, do it
        if (not CompactedWithCurrent.empty()) {
          Changed = true;

          // New artificial node representing an element of the compacted array
          auto *New = TS.createArtificialLayoutType();
          New->Size = Stride - AvailableSlack;

          // Helper lambda to compact the various components into the compacted
          // array.
          auto Compact = [&](const NeighborIterator &ToCompactIt) {
            auto &[TargetNode, EdgeTag] = *ToCompactIt;
            uint64_t OldOffset = EdgeTag->getOffsetExpr().Offset;
            revng_assert(OldOffset >= ArrayStartOffset);
            uint64_t OffsetInArray = (OldOffset - ArrayStartOffset) % Stride;
            TS.addInstanceLink(New,
                               TargetNode,
                               OffsetExpression{ OffsetInArray });
            return TS.eraseEdge(Parent, ToCompactIt);
          };

          // Compact all the array components.
          for (const NeighborIterator &ToCompact :
               CompactedWithCurrent.takeVector())
            Compact(ToCompact);
          // We compact ChildEdgeIt as last, so that it updates ChildEdgeNext
          // properly to continue the outer iteration.
          ChildEdgeNext = Compact(ChildEdgeIt);

          OffsetExpression NewStridedOffset{ ArrayStartOffset };
          NewStridedOffset.Strides.push_back(Stride);
          NewStridedOffset.TripCounts.push_back(TripCount);
          TS.addInstanceLink(Parent, New, std::move(NewStridedOffset));
        }
      }
    }
  }

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());

  return Changed;
}

} // end namespace dla
