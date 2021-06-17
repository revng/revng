//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include <algorithm>
#include <compare>
#include <numeric>
#include <tuple>
#include <type_traits>

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"

#include "revng/Support/Debug.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"

#include "../DLAHelpers.h"
#include "DLAStep.h"

namespace dla {

using LTSN = LayoutTypeSystemNode;

bool ComputeNonInterferingComponents::runOnTypeSystem(LayoutTypeSystem &TS) {
  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG() and TS.verifyInheritanceTree());

  bool Changed = false;

  // Helper set, to prevent visiting a node from multiple entry points.
  std::set<const LTSN *> Visited;

  for (LTSN *Root : llvm::nodes(&TS)) {
    revng_assert(Root != nullptr);
    if (not isRoot(Root))
      continue;

    for (LTSN *N : llvm::post_order_ext(Root, Visited)) {
      revng_assert(not isLeaf(N) or hasValidLayout(N));
      revng_assert(N->Size);

      struct OrderedChild {
        int64_t Offset;
        decltype(N->Size) Size;
        LTSN *Child;
        // Make it sortable
        std::strong_ordering operator<=>(const OrderedChild &) const = default;
      };
      using ChildrenVec = llvm::SmallVector<OrderedChild, 8>;
      using OrderedChildIt = ChildrenVec::iterator;

      // Collect the children in a vector. Here we use the OrderedChild struct,
      // that embeds info on the size and offset of the children, so that we can
      // later sort the vector according to it.
      ChildrenVec Children;
      bool InheritsFromOther = false;
      for (auto &[Child, EdgeTag] : llvm::children_edges<LTSN *>(N)) {

        auto OrdChild = OrderedChild{
          /* .Offset */ 0LL,
          /* .Size   */ Child->Size,
          /* .Child  */ Child,
        };

        switch (EdgeTag->getKind()) {

        case TypeLinkTag::LK_Instance: {
          const OffsetExpression &OE = EdgeTag->getOffsetExpr();
          revng_assert(OE.Offset >= 0LL);
          revng_assert(OE.Strides.size() == OE.TripCounts.size());

          OrdChild.Offset = OE.Offset;

          for (const auto &[TripCount, Stride] :
               llvm::reverse(llvm::zip(OE.TripCounts, OE.Strides))) {

            revng_assert(Stride > 0LL);
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
          InheritsFromOther = true;
        } break;

        default:
          revng_unreachable("unexpected edge tag");
        }

        revng_assert(OrdChild.Offset >= 0LL and OrdChild.Size > 0ULL);
        Children.push_back(std::move(OrdChild));
      }

      // If there are no children, there's nothing to do. There might be some
      // accesses performed directly from N, but they always interfere with each
      // other (because they start at the same base address), so they always
      // constitute a single non-interfering component and we can leave them
      // alone.
      if (Children.empty()) {
        N->InterferingInfo = AllChildrenAreInterfering;
        continue;
      }

      // If there is only one children and no accesses, we are sure that there's
      // nothing to do, because the only children cannot interfere with anything
      // else, and it is already a component on its own.
      auto NumAccesses = N->AccessSizes.size();
      if (Children.size() == 1ULL and not NumAccesses) {
        N->InterferingInfo = AllChildrenAreNonInterfering;
        continue;
      }

      // Sort the children. Thanks to the ordering of std::tuple, children at
      // lower offsets will be sorted before children with higher offsets, and
      // for children at the same offset, the smaller will be sorted before the
      // larger ones.
      std::sort(Children.begin(), Children.end());

      // Struct that represents a non-interfering component.
      // StartChildIt and EndChildIt are iterators into Children.
      // StartByte and EndByte are computed during the identification.
      // They are necessary for the creation of the artificial children in the
      // type system graph later.
      // NumChildren is the number of children or accesses that contribute to
      // the Component.
      // HasAccesses is true is this Component includes the accesses.
      struct Component {
        OrderedChildIt StartChildIt;
        OrderedChildIt EndChildIt;
        int64_t StartByte;
        uint64_t EndByte;
        size_t NumChildren;
        bool HasAccesses;
      };

      llvm::SmallVector<Component, 8> Components;
      {
        // Helper lambda to create a new component starting from the iterator to
        // a children that becomes the first element of the component.
        const auto MakeNewComponentFromChild = [](OrderedChildIt ChildIt) {
          auto ChildBeginByte = ChildIt->Offset;
          auto ChildEndByte = ChildBeginByte + ChildIt->Size;
          return Component{
            /* .StartChildIt */ ChildIt,
            /* .EndChildIt   */ std::next(ChildIt),
            /* .StartByte    */ ChildBeginByte,
            /* .EndByte      */ ChildEndByte,
            /* .NumChildren  */ 1ULL,
            /* .HasAccesses  */ false,
          };
        };

        OrderedChildIt ChildIt = Children.begin();
        {
          auto FirstChildComp = MakeNewComponentFromChild(ChildIt);

          if (NumAccesses) {
            int64_t AccessStartByte = 0LL;
            auto MaxIt = std::max_element(N->AccessSizes.begin(),
                                          N->AccessSizes.end());
            uint64_t AccEndByte = MaxIt != N->AccessSizes.end() ? *MaxIt : 0ULL;

            revng_assert(FirstChildComp.StartByte >= 0);
            if (static_cast<uint64_t>(FirstChildComp.StartByte) < AccEndByte) {
              // Accesses interfere with the first component.
              // Update the current component to reflect it.
              FirstChildComp.StartByte = AccessStartByte;
              FirstChildComp.EndByte = std::max(FirstChildComp.EndByte,
                                                AccEndByte);
              FirstChildComp.NumChildren += NumAccesses;
              FirstChildComp.HasAccesses = true;
            } else {
              // Accesses are present, but they don't interfere with the node
              // children, so we can create a separate non-interfering
              // components just for them.
              Components.push_back(Component{
                /* .StartChildIt */ ChildIt,
                /* .EndChildIt   */ ChildIt,
                /* .StartByte    */ AccessStartByte,
                /* .EndByte      */ AccEndByte,
                /* .NumChildren  */ NumAccesses,
                /* .HasAccesses  */ true,
              });
            }
          }
          Components.push_back(std::move(FirstChildComp));
        }

        OrderedChildIt ChildEnd = Children.end();
        while (++ChildIt != ChildEnd) {

          auto &CurrComp = Components.back();
          revng_assert(CurrComp.StartByte >= 0);

          auto CompStartByte = static_cast<uint64_t>(CurrComp.StartByte);
          revng_assert(CompStartByte < CurrComp.EndByte);

          const auto &[ChildStartByte, ChildSize, _] = *ChildIt;
          revng_assert(ChildStartByte >= 0 and ChildSize > 0);

          auto ChildBeginByte = static_cast<uint64_t>(ChildStartByte);
          revng_assert(ChildBeginByte >= CompStartByte);

          if (ChildBeginByte >= CurrComp.EndByte) {
            // The next candidate child falls entirely past the end of the
            // component that we've been accumulating until now.
            // Create a new compoenent and push it into Components.
            Components.push_back(MakeNewComponentFromChild(ChildIt));
          } else {
            // The next candidate child interferes with the current component,
            // so it must be part of it.
            // Make sure that we update the EndByte.
            CurrComp.EndByte = std::max(CurrComp.EndByte,
                                        ChildBeginByte + ChildSize);
            CurrComp.EndChildIt = std::next(ChildIt);
            ++(CurrComp.NumChildren);
          }
        }
      }

      // If we have less than two components there's nothing to do.
      if (Components.size() < 2) {
        N->InterferingInfo = AllChildrenAreInterfering;
        continue;
      }

      // Helper lambda to filter the Components with more than one element.
      // We don't care about Components with 0 or 1 elements because they don't
      // need to be changed, because they are already non-interfering.
      const auto HasManyElements = [](const Component &C) {
        return C.NumChildren > 1ULL;
      };

      // For each Component with more than one element we have to create a new
      // node in the type system, and push the edges from N to the elements of
      // the component down to the newly created node.
      bool FoundAccesses = false;
      for (auto &C : llvm::make_filter_range(Components, HasManyElements)) {
        Changed = true;

        // Create the node representing the component
        LTSN *New = TS.createArtificialLayoutType();
        New->InterferingInfo = AllChildrenAreInterfering;

        // Set its size to the size of the component
        revng_assert(C.StartByte >= 0);
        revng_assert(C.EndByte > static_cast<uint64_t>(C.StartByte));
        New->Size = C.EndByte - static_cast<uint64_t>(C.StartByte);

        // Move edges that were going directly from N to the children in the
        // component C, so that these edges now go from New to Child.
        // This effectively disconnects N from its children that are part of C.
        // Those children will have New instead of N as predecessor.
        // While moving the edges, the offset on the edge is updated.
        using llvm::iterator_range;
        auto OrderedChildRange = iterator_range(C.StartChildIt, C.EndChildIt);
        for (auto &OrderedChild : OrderedChildRange)
          TS.moveEdges(N, New, OrderedChild.Child, -C.StartByte);

        // If the component C includes the accesses we need to move the
        // accessess down to New.
        if (C.HasAccesses) {
          revng_assert(not FoundAccesses);
          FoundAccesses = true;
          revng_assert(not C.StartByte);
          New->AccessSizes = std::move(N->AccessSizes);
        }

        // Add a link between N and the New node representing the component.
        // The component is at offset C.StartByte inside N.
        // If this offset is zero we add an inheritance edge, otherwise an
        // instance edge.
        if (C.StartByte)
          TS.addInstanceLink(N, New, OffsetExpression(C.StartByte));
        else
          TS.addInheritanceLink(N, New);
      }

      N->InterferingInfo = AllChildrenAreNonInterfering;
    }
  }

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG() and TS.verifyInheritanceTree());

  return Changed;
}

} // end namespace dla
