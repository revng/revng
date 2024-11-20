//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/DataLayoutAnalysis/DLATypeSystem.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#include "FieldSizeComputation.h"

uint64_t getFieldSize(const dla::LayoutTypeSystemNode *Child,
                      const dla::TypeLinkTag *EdgeTag) {
  uint64_t Result = 0ULL;

  auto ChildSize = Child->Size;
  revng_assert(ChildSize > 0ULL);

  switch (EdgeTag->getKind()) {

  case dla::TypeLinkTag::LK_Instance: {
    const dla::OffsetExpression &OE = EdgeTag->getOffsetExpr();
    revng_assert(OE.Strides.size() == OE.TripCounts.size());

    // Ignore stuff at negative offsets.
    revng_assert(OE.Offset >= 0ULL);

    // If we have an array, we have to compute its size, taking into
    // account the strides and the trip counts.
    // In an OffsetExpression the larger strides come first, so we need to build
    // the instance bottom-up, in reverse, starting from the smaller strides.
    uint64_t PrevStride = 0ULL;
    for (const auto &[TripCount, Stride] :
         llvm::reverse(llvm::zip(OE.TripCounts, OE.Strides))) {

      revng_assert(not PrevStride or PrevStride < Stride);
      revng_assert(Stride > 0ULL);
      auto StrideSize = Stride;

      // If we have a TripCount, we expect it to be strictly positive.
      revng_assert(not TripCount.has_value() or TripCount.value() > 0ULL);

      // Arrays with unknown numbers of elements are considered as if
      // they had a single element
      auto NumElems = TripCount.has_value() ? TripCount.value() : 1;
      revng_assert(NumElems);

      // Here we are computing the larger size that is known to be
      // accessed. So if we have an array, we consider it to be one
      // element shorter than expected, and we add ChildSize only once
      // at the end.
      // This is equivalent to:
      // ChildSize = (NumElems * StrideSize) - (StrideSize - ChildSize);
      ChildSize = ((NumElems - 1) * StrideSize) + ChildSize;
    }

    revng_assert(ChildSize);
    Result = ChildSize;
  } break;

  default:
    revng_unreachable("unexpected edge");
  }

  return Result;
}

uint64_t getFieldUpperMember(const dla::LayoutTypeSystemNode *Child,
                             const dla::TypeLinkTag *EdgeTag) {
  auto ChildSize = getFieldSize(Child, EdgeTag);
  revng_assert(ChildSize > 0ULL);

  uint64_t ChildOffset = 0ULL;
  if (EdgeTag->getKind() == dla::TypeLinkTag::LK_Instance) {
    const dla::OffsetExpression &OE = EdgeTag->getOffsetExpr();
    ChildOffset = std::max<int64_t>(OE.Offset, ChildOffset);
  }
  return ChildOffset + ChildSize;
}
