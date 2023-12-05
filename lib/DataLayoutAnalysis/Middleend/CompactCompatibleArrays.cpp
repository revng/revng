//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include <compare>
#include <functional>
#include <optional>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"

#include "DLAStep.h"

static uint64_t
getTripCount(uint64_t StartOffset, uint64_t EndOffset, uint64_t Stride) {
  revng_assert(EndOffset > StartOffset);
  auto DivRem = std::lldiv(EndOffset - StartOffset, Stride);
  return DivRem.quot + (DivRem.rem ? 1ULL : 0ULL);
}

namespace dla {

using NeighborIterator = LayoutTypeSystem::NeighborIterator;

// Helper ordering for NeighborIterators. We need it here because we need to use
// such iterators and keys in associative containers, and we want neighbors with
// lower offset to come first.
// Notice that this might have undefined behavior if dereferencing either LHS
// or RHS is undefined behavior itself.
// The bottom line is that we should never insert invalid iterators into
// associative containers.
static std::weak_ordering operator<=>(const NeighborIterator &LHS,
                                      const NeighborIterator &RHS) {
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

/// Simple struct representing all the information we need to track on an array
/// to be created compacting many strided instance edges.
struct CompactedArrayInfo {
  // The start offset of the array inside the struct that owns it.
  uint64_t StartOffset;

  // The end offset of the array inside the struct that owns it.
  uint64_t EndOffset;

  // The stride of the array.
  uint64_t Stride;

  // 1 + the highest offset inside the array element for which DLA can guarantee
  // that it can see a memory access.
  // Should always be > 0 and <= Stride.
  uint64_t AccessedElementSize;

  // The number of bytes available outside the array, in the owning struct,
  // before the array itself.
  // It is an upper bound on how much it is possible to decrease the StartOffset
  // of the array.
  // It's always required to be at most (Stride - AccessedElementSize),
  // otherwise it would be possible to decrease the StartOffset to "shift" the
  // array in a way that would leave part of the AccessedElementSize out.
  uint64_t Slack;

  bool operator==(const CompactedArrayInfo &) const = default;
  std::strong_ordering operator<=>(const CompactedArrayInfo &) const = default;
};

CompactedArrayInfo makeCompactedArrayInfo(const auto &ArrayEdge) {
  revng_assert(isStridedInstance(ArrayEdge));

  const auto &[Child, OE] = ArrayEdge;
  const auto &[Offset, Strides, TripCounts] = OE->getOffsetExpr();
  revng_assert(Strides.size() == 1ULL);
  revng_assert(TripCounts.size() == 1ULL);

  uint64_t ChildSize = Child->Size;
  revng_assert(ChildSize > 0ULL);

  uint64_t Stride = Strides.front();
  revng_assert(Stride >= ChildSize);

  uint64_t TripCount = TripCounts.front().value_or(1ULL);
  revng_assert(TripCount > 0ULL);

  uint64_t EndOffset = Offset + (Stride * (TripCount - 1ULL)) + ChildSize;

  uint64_t Slack = std::min(Offset, Stride - ChildSize);

  return CompactedArrayInfo{
    .StartOffset = Offset,
    .EndOffset = EndOffset,
    .Stride = Stride,
    .AccessedElementSize = ChildSize,
    .Slack = Slack,
  };
}

/// Tries to adjust ToShift slightly moving it to lower StartOffset, so that
/// its element is aligned with AlignTo's element.
/// If this is successful, it then tries to compact the two resulting arrays, to
/// create a larger array that contains both the shifted ToShift, and AlignTo.
static std::optional<CompactedArrayInfo>
tryShiftLowerAndCompact(const CompactedArrayInfo &ToShift,
                        const CompactedArrayInfo &AlignTo) {

  revng_assert(ToShift.Stride == AlignTo.Stride);
  uint64_t Stride = ToShift.Stride;

  // Compute the RequiredSlack, i.e. the number of bytes we have to shift
  // ToShift towards a lower StartOffset, to align its element with ToAlign's
  // element.
  uint64_t RequiredSlack = 0ULL;
  {
    // Consider AlignTo's element aligned to 0, because that's the baseline we
    // want to align ToShift to.

    uint64_t ToShiftAlignment = ToShift.StartOffset % Stride;
    uint64_t ToMatchAlignment = AlignTo.StartOffset % Stride;
    if (ToShiftAlignment > ToMatchAlignment) {
      // If ToShiftAlignment is larger than ToMatchAlignment, we just shift it
      // down by ToMatchAlignment.
      RequiredSlack = ToShiftAlignment - ToMatchAlignment;
    } else if (ToShiftAlignment < ToMatchAlignment) {
      // Otherwise, we have to add Stride first, otherwise we would underflow.
      RequiredSlack = ToShiftAlignment + Stride - ToMatchAlignment;
    }
  }
  revng_assert(RequiredSlack < Stride);

  // So now the two elements of the arrays are aligned like this:
  // AlignTo |<------------Stride------------>|
  // ToShift |<--RequiredSlack-->|<------------Stride------------>|

  // Now we'd like to try to shift ToShift to a lower starting offset, so that
  // its starting point matches the starting point of AlignTo.
  // We can only do that if ToShift has enough Slack.
  if (ToShift.Slack < RequiredSlack)
    return std::nullopt;

  // Now we're shifting ToShift towards lower starting addresses, so we have to
  // decrease the StartOffset accordingly.
  // We can assume that ToShift.StartOffset is always larger or equal to
  // RequiredSlack because of how ToShift was generated and because of the fact
  // that StartOffset is part of the computation used to determine the value of
  // ToShift.Slack.
  // Also, AlignTo could start at a lower offset than ToShift, so we actually
  // have to take the minimum.
  revng_assert(ToShift.StartOffset >= RequiredSlack);
  uint64_t NewStartOffset = std::min(ToShift.StartOffset - RequiredSlack,
                                     AlignTo.StartOffset);

  // We can obtain the NewSlack by subtracting the RequiredSlack from
  // ToShift.Slack. This is not enough though, because AlignTo.Slack could be
  // even smaller, so we have to take the minimum.
  uint64_t NewSlack = std::min(ToShift.Slack - RequiredSlack, AlignTo.Slack);

  // We can obtain the NewElement size by adding the RequiredSlack to
  // ToShift.AccessedElementSize. Again, this is not enough, because
  // AlignTo.AccessedElementSize could be even larger, so we have to take the
  // maximum.
  revng_assert(ToShift.AccessedElementSize <= Stride - RequiredSlack);
  uint64_t NewElementSize = std::max(ToShift.AccessedElementSize
                                       + RequiredSlack,
                                     AlignTo.AccessedElementSize);

  // Compute the new TripCount.
  // This a little bit tricky, because it might be tempting to just say that
  // we can compute the new EndOffset as max(ToShift.EndOffset,
  // AlignTo.EndOffset) and then compute the TripCount as
  // getTripCount(NewStartOffset, EndOffset, Stride).
  // This is wrong when the array with higher EndOffset also has a larger Slack
  // (or a smaller AccessedElementSize, which in this case should be
  // equivalent).
  // The correct way to compute this is to compute the new TripCount as the max
  // of the TripCounts to go from NewStartOffset to both ToShift.EndOffset and
  // ToFit.EndOffset, that are never changed by previous computations.
  // Then, with this new TripCount we can compute the new EndOffset considering
  // the new AccessedElementSize and the Stride.
  uint64_t TripCountA = getTripCount(NewStartOffset, ToShift.EndOffset, Stride);
  uint64_t TripCountB = getTripCount(NewStartOffset, AlignTo.EndOffset, Stride);
  uint64_t TripCount = std::max(TripCountA, TripCountB);

  // The common EndOffset is the highest among the two.
  uint64_t NewEndOffset = NewStartOffset + (Stride * (TripCount - 1ULL))
                          + NewElementSize;

  // If the new EndOffset is larger than the both the EndOffsets we're trying to
  // compact, this compaction would introduce stuff to the right of what we've
  // seen in the binary.
  // This is something we're not allowed to do, because it might mess up the
  // size of the containing struct.
  if (NewEndOffset > std::max(ToShift.EndOffset, AlignTo.EndOffset))
    return std::nullopt;

  return CompactedArrayInfo{
    .StartOffset = NewStartOffset,
    .EndOffset = NewEndOffset,
    .Stride = Stride,
    .AccessedElementSize = NewElementSize,
    .Slack = NewSlack,
  };
}

static std::optional<CompactedArrayInfo> &
pickBest(std::optional<CompactedArrayInfo> &MaybeA,
         std::optional<CompactedArrayInfo> &MaybeB) {

  if (not MaybeA.has_value())
    return MaybeB;

  if (not MaybeB.has_value())
    return MaybeA;

  // Here both inputs have a value
  CompactedArrayInfo &A = MaybeA.value();
  CompactedArrayInfo &B = MaybeB.value();

  // Should have the same Stride
  revng_assert(A.Stride == B.Stride);

  // Should be aligned
  revng_assert((A.StartOffset % A.Stride) == (B.StartOffset % B.Stride));

  // Should end in the same place.
  // If they didn't, at least one of them should have increased the maximum
  // EndOffset among the two of them, which should be filtered away.
  revng_assert(A.EndOffset == B.EndOffset);

  // If they don't start at the same StartOffset, the one with the highest
  // StartOffset should be picked as best, because it generates less data at the
  // beginning.
  if (auto Cmp = A.StartOffset <=> B.StartOffset; Cmp != 0)
    return (Cmp < 0) ? MaybeB : MaybeA;

  // Because of how the two candidates are constructed we should have the
  // guarantee that if the new start offsets are the same, then both Slack and
  // AccessedElementSize are the same.
  revng_assert(A == B);

  return MaybeA;
}

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
        CompactedArrayInfo Current = makeCompactedArrayInfo(ArrayEdge);

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

          CompactedArrayInfo Sibling = makeCompactedArrayInfo(ArraySiblingEdge);

          // If the Sibling has a mismatching stride we don't compact it.
          if (Sibling.Stride != Current.Stride)
            continue;

          // If the Sibling starts after the Current ends, they don't overlap
          // and we don't compact them.
          if (Sibling.StartOffset >= Current.EndOffset)
            continue;

          // If the Sibling ends before the Current starts, they don't overlap
          // and we dont compact them.
          if (Sibling.EndOffset <= Current.StartOffset)
            continue;

          // Here we have a strong evidence that Sibling overlaps the range
          // of Current for at least one byte.
          // At this point we want to find out if we can compact them.

          // First of all we have to try and adjust the alignment of the array
          // element, so that the elements of Current and Sibling are aligned.
          // We can only do this by trying to move the start of an array element
          // to a lower offset, but we have to try both to move Current and to
          // move Sibling, and see which one gives the best results.

          // Try to shift Current to a lower StartOffset, to match Sibling's
          // alignment.
          std::optional<CompactedArrayInfo>
            FittedShiftingCurrent = tryShiftLowerAndCompact(Current, Sibling);

          // Try to shift Sibling to a lower StartOffset, to match Current's
          // alignment.
          std::optional<CompactedArrayInfo>
            FittedShiftingSibling = tryShiftLowerAndCompact(Sibling, Current);

          std::optional<CompactedArrayInfo>
            &MaybeBest = pickBest(FittedShiftingCurrent, FittedShiftingSibling);

          // If both adjustments failed, we give up on trying to compact Current
          // and Sibling, and skip to the next sibling.
          if (not MaybeBest.has_value())
            continue;

          CompactedArrayInfo &Best = MaybeBest.value();

          // If we're moving the StartOffset backward (or the EndOffset
          // forward) the size of the range we're tracking is going to increase.
          // We have to restart looking at array siblings because, given that
          // the range has increased, there could be new siblings that have
          // overlapping bytes, and we have to take them into consideration too.
          if (Best.StartOffset < Current.StartOffset
              or Best.EndOffset > Current.EndOffset)
            SiblingEdgeNext = GT::child_edge_begin(Parent);

          Current = Best;

          // Here we know that ArraySibling can be compacted with the current
          // array we're tracking.
          CompactedWithCurrent.insert(SiblingEdgeIt);
        }

        // TODO: we should think if it's beneficial to incorporate single
        // elements, and how to treat arrays with unknown trip counts

        // If we have something to compact, do it
        if (not CompactedWithCurrent.empty()) {
          Changed = true;

          // New artificial node representing an element of the compacted array
          auto *New = TS.createArtificialLayoutType();
          New->Size = Current.AccessedElementSize;

          // Helper lambda to compact the various components into the compacted
          // array.
          auto Compact = [&](const NeighborIterator &ToCompactIt) {
            auto &[TargetNode, EdgeTag] = *ToCompactIt;
            uint64_t OldOffset = EdgeTag->getOffsetExpr().Offset;
            revng_assert(OldOffset >= Current.StartOffset);
            uint64_t OffsetInArray = (OldOffset - Current.StartOffset)
                                     % Current.Stride;
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

          OffsetExpression NewStridedOffset{ Current.StartOffset };
          NewStridedOffset.Strides.push_back(Current.Stride);
          NewStridedOffset.TripCounts
            .push_back(getTripCount(Current.StartOffset,
                                    Current.EndOffset,
                                    Current.Stride));
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
