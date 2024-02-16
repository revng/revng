//
// This file is distributed under the MIT License. See LICENSE.md for details.
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

#include "DLAStep.h"
#include "FieldSizeComputation.h"

namespace dla {

using LTSN = LayoutTypeSystemNode;
using GraphNodeT = LTSN *;
using NonPointerFilterT = EdgeFilteredGraph<GraphNodeT, isNotPointerEdge>;

bool ComputeNonInterferingComponents::runOnTypeSystem(LayoutTypeSystem &TS) {
  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());

  bool Changed = false;

  // Helper set, to prevent visiting a node from multiple entry points.
  std::set<const LTSN *> Visited;

  for (LTSN *Root : llvm::nodes(&TS)) {
    revng_assert(Root != nullptr);
    if (not isRoot(Root))
      continue;

    for (LTSN *N : llvm::post_order_ext(NonPointerFilterT(Root), Visited)) {
      revng_assert(N->Size);

      struct OrderedChild {
        dla::LayoutTypeSystemNode::NeighborsSet::iterator ChildIt;
        size_t FieldSize;

        // Make it sortable with a different order
        std::strong_ordering operator<=>(const OrderedChild &Other) const {
          auto &ThisEdgeTag = *ChildIt->second;
          auto &OtherEdgeTag = *Other.ChildIt->second;

          // Stuff that starts earlier goes first
          if (auto Cmp = ThisEdgeTag <=> OtherEdgeTag; 0 != Cmp)
            return Cmp;

          // Smaller stuff goes first
          if (auto Cmp = FieldSize <=> Other.FieldSize; 0 != Cmp)
            return Cmp;

          // Finally sort by address
          return ChildIt->first <=> Other.ChildIt->first;
        }

        auto getBeginEndByte() const {
          auto ChildBeginByte = ChildIt->second->getOffsetExpr().Offset;
          auto ChildEndByte = ChildBeginByte + FieldSize;
          return std::make_pair(ChildBeginByte, ChildEndByte);
        }
      };
      using ChildrenVec = llvm::SmallVector<OrderedChild, 8>;
      using OrderedChildIt = ChildrenVec::iterator;

      // Collect the children in a vector. Here we use the OrderedChild struct,
      // that has a dedicated <=> operator so that we can later sort the vector
      // according to it.
      ChildrenVec Children;
      auto NChildIt = N->Successors.begin();
      auto NChildEnd = N->Successors.end();
      for (; NChildIt != NChildEnd; ++NChildIt) {
        if (isPointerEdge(*NChildIt))
          continue;

        Children.push_back(OrderedChild{
          .ChildIt = NChildIt,
          .FieldSize = getFieldSize(NChildIt->first, NChildIt->second),
        });
      }

      // If there are no children, there's nothing to do. There might be some
      // accesses performed directly from N, but they always interfere with each
      // other (because they start at the same base address), so they always
      // constitute a single non-interfering component and we can leave them
      // alone.
      if (Children.empty()) {
        N->InterferingInfo = AllChildrenAreNonInterfering;
        continue;
      }

      // If there is only one children and no accesses, we are sure that there's
      // nothing to do, because the only children cannot interfere with anything
      // else, and it is already a component on its own.
      if (Children.size() == 1ULL) {
        N->InterferingInfo = AllChildrenAreNonInterfering;
        continue;
      }

      // Sort the children. Thanks to the ordering of OrderedChild, children at
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
        uint64_t StartByte;
        uint64_t EndByte;
        size_t NumChildren;
        bool HasAccesses;
      };

      llvm::SmallVector<Component, 8> Components;
      {
        // Helper lambda to create a new component starting from the iterator to
        // a children that becomes the first element of the component.
        const auto MakeNewComponentFromChild = [](OrderedChildIt ChildIt) {
          const auto &[ChildBeginByte,
                       ChildEndByte] = ChildIt->getBeginEndByte();
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
        auto FirstChildComp = MakeNewComponentFromChild(ChildIt);
        Components.push_back(std::move(FirstChildComp));

        OrderedChildIt ChildEnd = Children.end();
        while (++ChildIt != ChildEnd) {

          auto &CurrComp = Components.back();
          revng_assert(CurrComp.StartByte >= 0);

          auto CompStartByte = static_cast<uint64_t>(CurrComp.StartByte);
          revng_assert(CompStartByte < CurrComp.EndByte);

          const auto &[ChildStartByte,
                       ChildEndByte] = ChildIt->getBeginEndByte();
          auto ChildSize = ChildEndByte - ChildStartByte;
          revng_assert(ChildStartByte >= 0 and ChildSize > 0);

          auto ChildBeginByte = static_cast<uint64_t>(ChildStartByte);
          revng_assert(ChildBeginByte >= CompStartByte);

          if (ChildBeginByte >= CurrComp.EndByte) {
            // The next candidate child falls entirely past the end of the
            // component that we've been accumulating until now.
            // Create a new component and push it into Components.
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
        revng_assert(not Components.empty());
        if (Components.back().NumChildren > 1)
          N->InterferingInfo = AllChildrenAreInterfering;
        else
          N->InterferingInfo = AllChildrenAreNonInterfering;
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
          TS.moveEdgeSource(N, New, OrderedChild.ChildIt, -C.StartByte);

        // Add a link between N and the New node representing the component.
        // The component is at offset C.StartByte inside N.
        TS.addInstanceLink(N, New, OffsetExpression(C.StartByte));
      }

      N->InterferingInfo = AllChildrenAreNonInterfering;
    }
  }

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());

  return Changed;
}

} // end namespace dla
