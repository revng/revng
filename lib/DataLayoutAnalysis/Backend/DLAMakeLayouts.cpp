//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/PostOrderIterator.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"

#include "DLAMakeLayouts.h"

using namespace llvm;

static Logger<> Log("dla-make-layouts");

namespace dla {

using LTSN = LayoutTypeSystemNode;
using GraphNodeT = LTSN *;
using NonPointerFilterT = EdgeFilteredGraph<GraphNodeT, isNotPointerEdge>;
using ConstNonPointerFilterT = EdgeFilteredGraph<const LTSN *,
                                                 isNotPointerEdge>;

static Layout *makeInstanceChildLayout(Layout *ChildType,
                                       const OffsetExpression &OE,
                                       LayoutVector &Layouts) {
  revng_assert(OE.Offset >= 0LL);

  // If we have trip counts we have an array of children of type ChildType,
  // otherwise ChildType already points to the right child type.
  revng_assert(OE.Strides.size() == OE.TripCounts.size());
  if (not OE.TripCounts.empty()) {
    Layout *Inner = ChildType;
    for (const auto &[TC, S] :
         llvm::reverse(llvm::zip(OE.TripCounts, OE.Strides))) {
      revng_assert(S > 0LL);
      Layout::layout_size_t StrideSize = (Layout::layout_size_t) (S);

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
        Layout *Padding = createLayout<PaddingLayout>(Layouts, PadSize);
        StructFields.push_back(Padding);
        Inner = createLayout<StructLayout>(Layouts, std::move(StructFields));
      }

      // Create the real array of Inner elements.
      Inner = createLayout<ArrayLayout>(Layouts, Inner, S, TC);
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
    StructFields.push_back(createLayout<PaddingLayout>(Layouts, Len));
    StructFields.push_back(ChildType);
    ChildType = createLayout<StructLayout>(Layouts, std::move(StructFields));
  }
  revng_assert(nullptr != ChildType);

  return ChildType;
}

static Layout *getLayout(const LayoutTypeSystem &TS,
                         LayoutPtrVector &OrderedLayouts,
                         const LTSN *N) {
  // First, find the node's equivalence class ID
  auto EqClassID = TS.getEqClasses().getEqClassID(N->ID);
  if (not EqClassID)
    return nullptr;

  revng_assert(*EqClassID < OrderedLayouts.size());
  // Get the layout at that position
  Layout *L = OrderedLayouts[*EqClassID];
  return L;
}

static Layout *makeLayout(const LayoutTypeSystem &TS,
                          const LTSN *N,
                          LayoutVector &Layouts,
                          LayoutPtrVector &OrderedLayouts) {
  switch (N->InterferingInfo) {

  case AllChildrenAreNonInterfering: {
    StructLayout::fields_container_t SFlds;

    // Create BaseLayout for leaf nodes
    revng_assert(not isLeaf(N) or N->Size);
    if (isLeaf(N)) {
      Layout *AccessLayout = createLayout<BaseLayout>(Layouts,
                                                      N->Size,
                                                      nullptr);

      // HACK: This needs to be done to distinguish pointer nodes before the
      // pointee layout is created
      if (llvm::any_of(N->Successors, isPointerEdge)) {
        BaseLayout *Base = llvm::cast<dla::BaseLayout>(AccessLayout);
        Base->PointeeLayout = AccessLayout;
      }

      return AccessLayout;
    }

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
    ChildrenVec Children;

    for (auto &[Child, EdgeTag] :
         llvm::children_edges<ConstNonPointerFilterT>(N)) {
      revng_log(Log, "Child ID: " << Child->ID);

      auto OrdChild = OrderedChild{
        /* .Offset */ 0LL,
        /* .Size   */ Child->Size,
        /* .Child  */ Child,
      };

      switch (EdgeTag->getKind()) {

      case TypeLinkTag::LK_Pointer:
        revng_abort("Only BaseLayouts are allowed to have pointer edges");

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

      default:
        revng_unreachable("unexpected edge tag");
      }

      if (OrdChild.Offset >= 0LL and OrdChild.Size > 0ULL) {
        Children.push_back(std::move(OrdChild));
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

    // For each member of the struct
    uint64_t CurSize = 0U;
    for (const auto &OrdChild : Children) {
      const auto &[StartByte, Size, Child] = OrdChild;
      revng_assert(StartByte >= 0LL and Size > 0ULL);
      uint64_t Start = static_cast<uint64_t>(StartByte);
      revng_assert(Start >= CurSize);
      auto PadSize = Start - CurSize;
      revng_assert(PadSize >= 0);

      // If there is a "hole" between accesses, add it as padding
      if (PadSize) {
        Layout *Padding = createLayout<PaddingLayout>(Layouts, PadSize);
        SFlds.push_back(Padding);
      }
      CurSize = Start + Size;

      Layout *ChildType = getLayout(TS, OrderedLayouts, Child);

      // Bail out if we have not constructed a union field, because it means
      // that this is not a supported case yet.
      if (not ChildType)
        return nullptr;

      SFlds.push_back(ChildType);
    }

    // This layout has no useful access or outgoing edges that can build the
    // type. Just skip it for now until we support handling richer edges and
    // emitting richer types
    if (SFlds.empty())
      return nullptr;

    Layout *CreatedLayout = createLayout<StructLayout>(Layouts, SFlds);

    return CreatedLayout;
  } break;

  case AllChildrenAreInterfering: {
    revng_assert(N->Successors.size() > 1);
    UnionLayout::elements_container_t UFlds;
    revng_assert(not isLeaf(N));

    // Look at all the instance-of edges and inheritance edges all together
    for (auto &[Child, EdgeTag] : children_edges<ConstNonPointerFilterT>(N)) {
      revng_log(Log, "Child ID: " << Child->ID);
      revng_assert(Child->Size);

      Layout *ChildType = getLayout(TS, OrderedLayouts, Child);

      // Ignore children for which we haven't created a layout, because they
      // only have children from which it was not possible to create valid
      // layouts.
      if (not ChildType) {
        revng_log(Log, "No corresponding layout for " << Child->ID);
        return nullptr;
      }

      switch (EdgeTag->getKind()) {

      case TypeLinkTag::LK_Instance: {
        revng_log(Log, "Instance");
        const OffsetExpression &OE = EdgeTag->getOffsetExpr();
        revng_log(Log, "Has Offset: " << OE.Offset);
        ChildType = makeInstanceChildLayout(ChildType, OE, Layouts);
      } break;

      default:
        revng_unreachable("unexpected edge");
      }

      // Bail out if we have not constructed a union field, because it means
      // that this is not a supported case yet.

      if (nullptr != ChildType) {
        bool New = UFlds.insert(ChildType).second;

        if (not New)
          revng_log(Log, "Duplicate layout found, size: " << UFlds.size());
      } else {
        revng_log(Log, "No type created for " << Child->ID);
      }
    }

    // This layout has no useful access or outgoing edges that can build the
    // type. Just skip it for now until we support handling richer edges and
    // emitting richer types
    if (UFlds.empty())
      return nullptr;

    // If a null layout was generated for one of the children, or the node has
    // more than one parent, the union might contain only one field. In this
    // case, there is no point in emitting a union, so a struct must be emitted.
    if (UFlds.size() == 1) {
      StructLayout::fields_container_t Fields;
      Fields.push_back(*UFlds.begin());
      return createLayout<StructLayout>(Layouts, Fields);
    }

    return createLayout<UnionLayout>(Layouts, UFlds);
  } break;

  case Unknown:
  default:
    revng_unreachable();
  }
  return nullptr;
}

static void connectPointersToPointees(const LayoutTypeSystem &TS,
                                      LayoutVector &Layouts,
                                      LayoutPtrVector &OrderedLayouts) {

  for (LTSN *N : llvm::nodes(&TS)) {
    revng_assert(N != nullptr);
    revng_log(Log, "Connecting " << N->ID);

    Layout *PointeeLayout = nullptr;
    // If it's a pointer, get the pointee's Layout
    using PtrFilterT = EdgeFilteredGraph<const LTSN *, isPointerEdge>;
    for (auto &[Child, EdgeTag] : llvm::children_edges<PtrFilterT>(N)) {
      // There can be at most one outgoing pointer edge
      revng_assert(PointeeLayout == nullptr);
      PointeeLayout = getLayout(TS, OrderedLayouts, Child);
    }

    if (PointeeLayout) {
      Layout *PointerLayout = getLayout(TS, OrderedLayouts, N);

      // HACK: This is needed in case the Pointer has been wrapped inside a
      // struct
      if (not isa<dla::BaseLayout>(PointerLayout)) {
        revng_assert(isa<dla::StructLayout>(PointerLayout));
        StructLayout *Wrapper = llvm::cast<dla::StructLayout>(PointerLayout);
        revng_assert(Wrapper->numFields() == 1);
        PointerLayout = *(Wrapper->fields().begin());
      }
      BaseLayout *Base = llvm::cast<dla::BaseLayout>(PointerLayout);
      Base->PointeeLayout = PointeeLayout;
    }
  }
}

LayoutPtrVector makeLayouts(const LayoutTypeSystem &TS, LayoutVector &Layouts) {
  if (Log.isEnabled())
    TS.dumpDotOnFile("final.dot");

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG() and TS.verifyUnions());

  // Prepare the vector of layouts that correspond to actual LayoutTypePtrs
  LayoutPtrVector OrderedLayouts;
  const auto EqClasses = TS.getEqClasses();
  OrderedLayouts.resize(EqClasses.getNumClasses());

  std::set<const LTSN *> Visited;

  // Create Layouts
  for (LTSN *Root : llvm::nodes(&TS)) {
    revng_assert(Root != nullptr);
    if (not isRoot(Root))
      continue;

    for (const LTSN *N : post_order_ext(NonPointerFilterT(Root), Visited)) {
      // Leaves need to have ValidLayouts, otherwise they should have been
      // trimmed by PruneLayoutNodesWithoutLayout
      revng_assert(not isLeaf(N) or N->Size);
      Layout *LN = makeLayout(TS, N, Layouts, OrderedLayouts);
      if (not LN) {
        revng_log(Log, "Node ID: " << N->ID << " Type: Empty");
        continue;
      }

      // Insert the layout at the index corresponding to the node's eq. class
      auto LayoutIdx = TS.getEqClasses().getEqClassID(N->ID);
      revng_assert(LayoutIdx);
      OrderedLayouts[*LayoutIdx] = LN;

      if (Log.isEnabled()) {
        llvm::dbgs() << "\nNode ID: " << N->ID << " Type: ";
        Layout::printText(llvm::dbgs(), LN);
        llvm::dbgs() << ";\n";
        Layout::printGraphic(llvm::dbgs(), LN);
        llvm::dbgs() << '\n';
      }
    }
  }

  connectPointersToPointees(TS, Layouts, OrderedLayouts);

  return OrderedLayouts;
};

ValueLayoutMap makeLayoutMap(const LayoutTypePtrVect &Values,
                             const LayoutPtrVector &Layouts,
                             const VectEqClasses &EqClasses) {
  ValueLayoutMap ValMap;

  for (size_t I = 0; I < Values.size(); I++) {
    // The layout of the I-th Value is stored at the EqClass(I) index
    auto LayoutIdx = EqClasses.getEqClassID(I);
    if (LayoutIdx and not Values[I].isEmpty()) {

      auto *L = Layouts[*LayoutIdx];
      if (not L)
        continue;

      auto NewPair = std::make_pair(Values[I], L);
      bool New = ValMap.insert(NewPair).second;
      revng_assert(New);
    }
  }

  return ValMap;
}
} // end namespace dla
