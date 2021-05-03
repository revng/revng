//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include <algorithm>
#include <compare>
#include <iterator>
#include <memory>
#include <set>
#include <string>
#include <type_traits>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/Support/Debug.h"

#include "revng-c/Decompiler/DLALayouts.h"

#include "DLAHelpers.h"
#include "DLAStep.h"
#include "DLATypeSystem.h"

using namespace llvm;

static Logger<> Log("dla-make-layouts");

namespace dla {

using LTSN = LayoutTypeSystemNode;

static Layout *makeInstanceChildLayout(Layout *ChildType,
                                       const OffsetExpression &OE,
                                       LayoutVector &Layouts) {
  revng_assert(OE.Offset >= 0LL);
  LayoutVector NewLayouts;

  // If we have trip counts we have an array of children of type ChildType,
  // otherwise ChildType already points to the right child type.
  revng_assert(OE.Strides.size() == OE.TripCounts.size());
  if (not OE.TripCounts.empty()) {
    Layout *Inner = ChildType;
    for (const auto &[TC, S] : llvm::zip(OE.TripCounts, OE.Strides)) {
      revng_assert(S > 0LL);
      Layout::layout_size_t StrideSize = (Layout::layout_size_t)(S);

      // For now, we don't handle stuff that for which the size of the element
      // is larger than the stride size
      if (StrideSize < Inner->size())
        return nullptr;

      // If the stride (StrideSize) is larger than the size of the inner
      // element, we need to reserve space after each element, using
      // padding.
      if (StrideSize > Inner->size()) {
        StructLayout::fields_container_t StructFields;
        StructFields.push_back(Inner);
        Layout::layout_size_t PadSize = StrideSize - Inner->size();
        Layout *Padding = createLayout<PaddingLayout>(NewLayouts, PadSize);
        StructFields.push_back(Padding);
        Inner = createLayout<StructLayout>(NewLayouts, std::move(StructFields));
      }

      // Create the real array of Inner elements.
      Inner = createLayout<ArrayLayout>(NewLayouts, Inner, S, TC);
    }
    ChildType = Inner;
  }
  revng_assert(nullptr != ChildType);

  if (OE.Offset > 0LL) {
    // Create padding to insert before the field, according to the
    // offset.
    ArrayLayout::length_t Len = OE.Offset;
    // Create the struct with the padding prepended to the field.
    StructLayout::fields_container_t StructFields;
    StructFields.push_back(createLayout<PaddingLayout>(NewLayouts, Len));
    StructFields.push_back(ChildType);
    ChildType = createLayout<StructLayout>(NewLayouts, std::move(StructFields));
  }
  revng_assert(nullptr != ChildType);

  Layouts.reserve(Layouts.size() + NewLayouts.size());
  for (auto &U : NewLayouts)
    Layouts.push_back(std::move(U));

  return ChildType;
}

static Layout *makeLayout(const LayoutTypeSystem &TS,
                          const LTSN *N,
                          std::map<const LTSN *, Layout *> &LayoutCTypes,
                          LayoutVector &Layouts) {

  revng_assert(not LayoutCTypes.count(N));
  LayoutVector NewLayouts;

  switch (N->InterferingInfo) {

  case AllChildrenAreNonInterfering: {

    auto NumAccesses = N->AccessSizes.size();
    uint64_t AccessSize = NumAccesses ? *N->AccessSizes.begin() : 0ULL;
    revng_assert(NumAccesses == 0 or NumAccesses == 1);

    StructLayout::fields_container_t SFlds;

    struct OrderedChild {
      int64_t Offset;
      decltype(N->Size) Size;
      LTSN *Child;
      // Make it sortable
      std::strong_ordering operator<=>(const OrderedChild &) const = default;
    };
    using ChildrenVec = llvm::SmallVector<OrderedChild, 8>;

    // Collect the children in a vector. Here we use the OrderedChild struct,
    // that embeds info on the size and offset of the children, so that we can
    // later sort the vector according to it.
    bool InheritsFromOther = false;
    ChildrenVec Children;
    for (auto &[Child, EdgeTag] : llvm::children_edges<const LTSN *>(N)) {

      auto OrdChild = OrderedChild{
        /* .Offset */ 0LL,
        /* .Size   */ Child->Size,
        /* .Child  */ Child,
      };

      switch (EdgeTag->getKind()) {

      case TypeLinkTag::LK_Instance: {
        const OffsetExpression &OE = EdgeTag->getOffsetExpr();
        revng_assert(OE.Strides.size() == OE.TripCounts.size());

        // Ignore stuff at negative offsets.
        if (OE.Offset < 0LL)
          continue;

        OrdChild.Offset = OE.Offset;
        for (const auto &[TripCount, Stride] :
             llvm::reverse(llvm::zip(OE.TripCounts, OE.Strides))) {

          // Strides should be positive. If they are not, we don't know
          // anything about how the children is layed out, so we assume the
          // children doesn't even exist.
          if (Stride <= 0LL) {
            OrdChild.Size = 0ULL;
            break;
          }

          auto StrideSize = static_cast<uint64_t>(Stride);

          // If we have a TripCount, we expect it to be strictly positive.
          revng_assert(not TripCount.has_value() or TripCount.value() > 0LL);

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
          OrdChild.Size = ((NumElems - 1) * StrideSize) + OrdChild.Size;
        }
      } break;

      case TypeLinkTag::LK_Inheritance: {
        revng_assert(not InheritsFromOther);
        // We can't have accesses, if we have inheritance, otherwise we'd have
        // that the inherited layout and the accesses do interfere with each
        // other, and we should have created a union, not a struct.
        revng_assert(not NumAccesses);
        InheritsFromOther = true;
      } break;

      default:
        revng_unreachable("unexpected edge tag");
      }

      if (OrdChild.Offset >= 0LL and OrdChild.Size > 0ULL) {
        Children.push_back(std::move(OrdChild));
        revng_assert(EdgeTag->getKind() != TypeLinkTag::LK_Instance
                     or not AccessSize
                     or static_cast<int64_t>(AccessSize) <= OrdChild.Offset);
      }
    }

    std::sort(Children.begin(), Children.end());

    if (VerifyLog.isEnabled()) {
      auto It = Children.begin();
      for (; It != Children.end() and std::next(It) != Children.end(); ++It) {
        int64_t ThisEndByte = It->Offset + static_cast<int64_t>(It->Size);
        revng_assert(ThisEndByte <= std::next(It)->Offset);
      }
    }

    revng_assert(not NumAccesses or NumAccesses == 1ULL);
    Layout *AccessLayout = nullptr;
    if (AccessSize)
      AccessLayout = createLayout<BaseLayout>(NewLayouts, AccessSize);

    bool First = true;

    for (const auto &OrdChild : Children) {
      const auto &[StartByte, Size, Child] = OrdChild;
      First = false;
      revng_assert(StartByte >= 0LL and Size > 0ULL);
      uint64_t Start = static_cast<uint64_t>(StartByte);
      revng_assert(Start >= AccessSize);
      auto PadSize = Start - AccessSize; // always >= 0;
      revng_assert(PadSize >= 0);
      if (PadSize) {
        Layout *Padding = createLayout<PaddingLayout>(NewLayouts, PadSize);
        SFlds.push_back(Padding);
      }
      AccessSize = Start + Size;

      revng_assert(LayoutCTypes.find(Child) != LayoutCTypes.end());
      Layout *ChildType = LayoutCTypes.at(Child);

      // Bail out if we have not constructed a union field, because it means
      // that this is not a supported case yet.
      revng_assert(nullptr != ChildType);
      SFlds.push_back(ChildType);
    }

    // This layout has no useful access or outgoing edges that can build the
    // type. Just skip it for now until we support handling richer edges and
    // emitting richer types
    if (SFlds.empty())
      return nullptr;

    Layout *CreatedLayout = (SFlds.size() > 1ULL) ?
                              createLayout<StructLayout>(NewLayouts, SFlds) :
                              *SFlds.begin();

    LayoutCTypes[N] = CreatedLayout;

    Layouts.reserve(Layouts.size() + NewLayouts.size());
    for (auto &U : NewLayouts)
      Layouts.push_back(std::move(U));

    return CreatedLayout;
  } break;

  case AllChildrenAreInterfering: {

    UnionLayout::elements_container_t UFlds;
    for (uint64_t AccessSize : N->AccessSizes) {
      revng_log(Log, "Access: " << AccessSize);
      UFlds.insert(createLayout<BaseLayout>(Layouts, AccessSize));
    }

    // Look at all the instance-of edges and inheritance edges all together
    bool InheritsFromOther = false;
    for (auto &[Child, EdgeTag] : children_edges<const LTSN *>(N)) {

      revng_log(Log, "Child ID: " << Child->ID);
      revng_assert(Child->Size);

      // Ignore children for which we haven't created a layout, because they
      // only have children from which it was not possible to create valid
      // layouts.
      auto ChildLayoutIt = LayoutCTypes.find(Child);
      revng_assert(ChildLayoutIt != LayoutCTypes.end());

      Layout *ChildType = ChildLayoutIt->second;

      switch (EdgeTag->getKind()) {

      case TypeLinkTag::LK_Instance: {
        revng_log(Log, "Instance");
        const OffsetExpression &OE = EdgeTag->getOffsetExpr();
        revng_log(Log, "Has Offset: " << OE.Offset);
        ChildType = makeInstanceChildLayout(ChildType, OE, NewLayouts);
      } break;

      case TypeLinkTag::LK_Inheritance: {
        revng_log(Log, "Inheritance");
        // Treated as instance at offset 0, but can only have one
        revng_assert(not InheritsFromOther);
        InheritsFromOther = true;
      } break;

      default:
        revng_unreachable("unexpected edge");
      }

      // Bail out if we have not constructed a union field, because it means
      // that this is not a supported case yet.
      if (nullptr != ChildType)
        UFlds.insert(ChildType);
    }

    // This layout has no useful access or outgoing edges that can build the
    // type. Just skip it for now until we support handling richer edges and
    // emitting richer types
    if (UFlds.empty())
      return nullptr;

    Layout *CreatedLayout = (UFlds.size() > 1ULL) ?
                              createLayout<UnionLayout>(NewLayouts, UFlds) :
                              *UFlds.begin();

    LayoutCTypes[N] = CreatedLayout;

    Layouts.reserve(Layouts.size() + NewLayouts.size());
    for (auto &U : NewLayouts)
      Layouts.push_back(std::move(U));
    return CreatedLayout;
  } break;

  case Unknown:
  default:
    revng_unreachable();
  }
  return nullptr;
}

static bool makeLayouts(const LayoutTypeSystem &TS,
                        LayoutVector &Layouts,
                        ValueLayoutMap &ValueLayouts) {
  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG() and TS.verifyInheritanceTree());

  std::map<const LTSN *, Layout *> LayoutCTypes;
  std::set<const LTSN *> Visited;
  for (LTSN *Root : llvm::nodes(&TS)) {
    revng_assert(Root != nullptr);
    if (not isRoot(Root))
      continue;

    for (const LTSN *N : post_order_ext(Root, Visited)) {
      // Leaves need to have ValidLayouts, otherwise they should have been
      // trimmed by PruneLayoutNodesWithoutLayout
      revng_assert(not isLeaf(N) or hasValidLayout(N));
      Layout *LN = makeLayout(TS, N, LayoutCTypes, Layouts);
      if (nullptr == LN) {
        revng_log(Log, "Node ID: " << N->ID << " Type: Empty");
        continue;
      }
      if (Log.isEnabled()) {
        llvm::dbgs() << "\nNode ID: " << N->ID << " Type: ";
        Layout::printText(llvm::dbgs(), LN);
        llvm::dbgs() << ";\n";
        Layout::printGraphic(llvm::dbgs(), LN);
        llvm::dbgs() << '\n';
      }
      if (auto *TypePtrs = TS.getLayoutTypePtrs(N)) {
        for (const auto &Value : *TypePtrs) {
          bool New = ValueLayouts.insert(std::make_pair(Value, LN)).second;
          revng_assert(New);
        }
      }
    }
  }
  return true;
};

bool MakeLayouts::runOnTypeSystem(LayoutTypeSystem &TS) {
  if (Log.isEnabled())
    TS.dumpDotOnFile("final.dot");
  return makeLayouts(TS, Layouts, ValueLayouts);
}

} // end namespace dla
