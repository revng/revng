//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/PostOrderIterator.h"

#include "revng/Support/Debug.h"

#include "DLAStep.h"
#include "FieldSizeComputation.h"

static Logger Log{ "dla-remove-stride-edges" };

namespace dla {

/// Check if an OffsetExpression has a StrideSize that is smaller that it's
/// inner element size.
static bool hasValidStrides(const LayoutTypeSystemNode::Link &Edge) {
  revng_assert(isInstanceEdge(Edge));
  const auto &[SuccNode, EdgeTag] = Edge;

  auto InnerSize = SuccNode->Size;
  // If InnerSize is zero, it means that it has become zero in a previous
  // iteration because we've removed so many invalid edges that the target of
  // this edge has become empty.
  if (not InnerSize)
    return false;

  const OffsetExpression &OE = EdgeTag->getOffsetExpr();
  revng_assert(OE.TripCounts.size() == OE.Strides.size());

  // In OE, by construction, strides go to from larger to smaller, so we need to
  // iterate in reverse to build check the strides bottom-up
  for (const auto &[TripCount, Stride] :
       llvm::reverse(llvm::zip(OE.TripCounts, OE.Strides))) {

    // If the Stride is smaller than the inner size the strided edge is not
    // iterating over an entire element, so it's not well formed.
    revng_assert(Stride > 0ULL);
    auto UStride = static_cast<uint64_t>(Stride);
    if (UStride < InnerSize)
      return false;

    uint64_t NumElems = TripCount.value_or(1);
    revng_assert(NumElems);
    InnerSize = (UStride * (NumElems - 1)) + InnerSize;
  }

  return true;
}

bool RemoveInvalidStrideEdges::runOnTypeSystem(LayoutTypeSystem &TS) {
  bool Changed = false;

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());

  std::set<const LayoutTypeSystemNode *> Visited;
  for (LayoutTypeSystemNode *Root : llvm::nodes(&TS)) {

    revng_log(Log, "Root ID: " << Root->ID);
    LoggerIndent Indent{ Log };

    if (not isInstanceRoot(Root)) {
      revng_log(Log, "NOT an Instance root");
      continue;
    }
    revng_log(Log, "Instance root");

    using InstanceNodeT = EdgeFilteredGraph<LayoutTypeSystemNode *,
                                            isInstanceEdge>;
    for (LayoutTypeSystemNode *N :
         llvm::post_order_ext(InstanceNodeT(Root), Visited)) {

      auto It = N->Successors.begin();
      auto End = N->Successors.end();
      auto Next = End;
      bool RemovedChild = false;
      for (; It != End; It = Next) {

        Next = std::next(It);

        if (not isInstanceEdge(*It))
          continue;

        if (hasValidStrides(*It))
          continue;

        // If we reach this point the edges has invalid strides, so we need to
        // remove it, taking care of the fact that it's bidirectional.
        LayoutTypeSystemNode *Succ = It->first;
        auto PredIt = Succ->Predecessors.find({ N, It->second });
        revng_assert(PredIt != Succ->Predecessors.end());
        Succ->Predecessors.erase(PredIt);
        N->Successors.erase(It);
        RemovedChild = true;
        Changed = true;
      }

      if (RemovedChild) {
        // If we reach this point, N had at least one outgoing instance edge
        // with strided access that was invalid.
        // Having an outgoing instance edge means that N is not a pointer node
        // nor a leaf node representing an access.
        // For this reason, given that we have remove a child, the size of N
        // could have changed, so we have to recompute from its children.
        uint64_t NewSize = 0ULL;

        // Look at all the instance-of edges and inheritance edges all together.
        using NonPointerNodeT = EdgeFilteredGraph<const LayoutTypeSystemNode *,
                                                  isNotPointerEdge>;
        revng_log(Log, "N's children");
        LoggerIndent MoreMoreIndent{ Log };
        for (auto &[Child, EdgeTag] :
             llvm::children_edges<NonPointerNodeT>(N)) {
          revng_log(Log, "Child->ID: " << Child->ID);
          revng_log(Log,
                    "EdgeTag->Kind: "
                      << dla::TypeLinkTag::toString(EdgeTag->getKind()));
          NewSize = std::max(NewSize, getFieldUpperMember(Child, EdgeTag));
        }

        N->Size = NewSize;
      }
    }
  }

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());

  return Changed;
}

} // end namespace dla
