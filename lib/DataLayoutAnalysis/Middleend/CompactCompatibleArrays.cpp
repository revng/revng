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

        // Here we're starting from ArrayEdge, which represents an array that
        // has not been grouped with others yet.

        // Now we want to look around ArrayEdge, and see if there are other
        // conflicting siblings that may need to be compacted with ArrayEdge
        // itself, possibly changing the trip count or the actual start of the
        // array.

        // Let's compute the running variables that we'll use to track where the
        // compacted array starts and ends.
        // These will be updated in flight until we finish the group and decide
        // all the siblings that need to be compacted with ArrayEdge.

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
        // bigger, if we find strong evidence in other siblings that the
        // compacted array should actually be larger.
        uint64_t ArrayEndOffset = InitialOffset + (Stride * (TripCount - 1ULL))
                                  + InitialChild->Size;

        // The initial slack inside an array element.
        // It represents how much we can move the boundaries of the array
        // element back, if we need to align it.
        // It represents the number of empty trailing bytes at the end of an
        // array element, but we can never move something more backwards than
        // the start of Parent, so we have to take ArrayStartOffset in
        // consideration.
        // The value of AvailableSlack always decreases during the iterations.
        revng_assert(Stride >= InitialChild->Size);
        uint64_t AvailableSlack = std::min(ArrayStartOffset,
                                           Stride - InitialChild->Size);

        // The offset of the first byte in the array element that we're not sure
        // that will be accessed.
        // It is always <= Stride - AvailableSlack;
        // Given that AvailableSlack only decreases and Stride is fixed,
        // AccessedElemSize only increases, possibly reaching Stride, but never
        // growing larger than that.
        uint64_t AccessedElemSize = InitialChild->Size;

        // Ok, now we start looking at other array siblings of ArrayEdge.
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
            // update the running values we're tracking.

            AvailableSlack -= RequiredSlack;
            ArrayStartOffset -= RequiredSlack;
            OffsetDifference -= RequiredSlack;

            // If we're moving the array start to the left we have to update the
            // size of the accessed element in the array.
            // It is always at least as large as the Sibling, and also at least
            // as large as the previous AccessedElemSize adjusted by the
            // RequiredSlack.
            AccessedElemSize = std::max(Sibling->Size,
                                        AccessedElemSize + RequiredSlack);

            // If there is still some OffsetDifference after adjusting the
            // slack, we have to add a whole bunch of new elements before
            // ArrayStartOffset.
            uint64_t NumPrecedingElements = OffsetDifference / Stride;
            if (NumPrecedingElements) {
              // Count how many elements we have to add before the array we're
              // tracking.

              // Effectively change the start of the range we're tracking as if
              // we added the proper amount of elements before the beginning. We
              // might need to subtract Stride once more (see below) if the
              // division above had non-zero remainder.
              ArrayStartOffset -= Stride * NumPrecedingElements;

              // Update the TripCount
              TripCount += NumPrecedingElements;
            }

            // Also, we're moving the ArrayStartOffset backward, so the the size
            // of the range we're tracking is going to increase, we have to
            // restart looking at array siblings because, given that the range
            // increases, there could be new siblings that have overlapping
            // bytes, and we have to take them into consideration too.
            if (RequiredSlack or NumPrecedingElements)
              SiblingEdgeNext = GT::child_edge_begin(Parent);

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
          if (OffsetOfSiblingInArray + Sibling->Size > Stride)
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
          // too. If RequiredSlack is not zero, we're moving the
          // ArrayStartOffset to the left, so we have to restart.
          if (RequiredSlack)
            SiblingEdgeNext = GT::child_edge_begin(Parent);
          revng_assert(ArrayStartOffset >= RequiredSlack);
          ArrayStartOffset -= RequiredSlack;

          // Update the available slack.
          // It may be smaller than AvailableSlack - RequiredSlack if
          // SiblingSize is large, because the compacted sibling will consume
          // more of the bytes in the Stride, effectively leaving less stride.
          AvailableSlack = std::min(AvailableSlack - RequiredSlack,
                                    Stride
                                      - (OffsetOfSiblingInArray
                                         + Sibling->Size));

          // Update the AccessedElemSize.
          // If we had a RequiredSlack it means that the array is being shifted
          // to the left, so the new AccessedElemSize is at least large like the
          // new Sibling, and at least large as the previous iteration adjusted
          // for the RequiredSlack.
          // If we didn't have a RequiredSlack, AccessedElemeSize stays the same
          // or at most it accesses the upper byte of the Sibling.
          if (RequiredSlack) {
            AccessedElemSize = std::max(Sibling->Size,
                                        AccessedElemSize + RequiredSlack);
          } else {
            AccessedElemSize = std::max(AccessedElemSize,
                                        OffsetOfSiblingInArray + Sibling->Size);
          }

          // Update the TripCount.
          // This is a little convoluted because we might have moved a little
          // bit the array start to the left and at the same time the sibling
          // array could end much later than the last access performed to the
          // array up until now.
          // So, basically starting from the new update ArrayStartOffset (that
          // takes into account the occasional shift to the left) we compute how
          // many trips we need to cover all the accesses that we know of,
          // namely represented by ArrayEndOffset and SiblingEndOffset (these
          // have not been updated yet, so they're still valid).
          // Then we take the largest of the two trip counts.
          auto OldDivRem = std::lldiv(ArrayEndOffset - ArrayStartOffset,
                                      Stride);
          auto NewDivRem = std::lldiv(SiblingEndOffset - ArrayStartOffset,
                                      Stride);
          uint64_t OldTripCount = OldDivRem.quot
                                  + (OldDivRem.rem ? 1ULL : 0ULL);
          uint64_t NewTripCount = NewDivRem.quot
                                  + (NewDivRem.rem ? 1ULL : 0ULL);
          TripCount = std::max(OldTripCount, NewTripCount);

          // Update the ArrayEndOffset, using the updated ArrayStartOffset, the
          // new updated TripCount and the newly updated AccessedElemSize.
          ArrayEndOffset = ArrayStartOffset + (Stride * (TripCount - 1ULL))
                           + AccessedElemSize;
        }

        // TODO: we should think if it's beneficial to incorporate single
        // elements, and how to treat arrays with unknown trip counts

        // If we have something to compact, do it
        if (not CompactedWithCurrent.empty()) {
          Changed = true;

          // New artificial node representing an element of the compacted array
          auto *New = TS.createArtificialLayoutType();
          New->Size = AccessedElemSize;

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
