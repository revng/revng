//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/PostOrderIterator.h"

#include "revng/Support/Debug.h"

#include "DLAStep.h"

static Logger<> Log{ "dla-remove-stride-edges" };

namespace dla {

/// Check if an OffsetExpression has a StrideSize that is smaller that it's
/// inner element size.
static bool hasValidStrides(const LayoutTypeSystemNode::Link &Edge) {
  revng_assert(isInstanceEdge(Edge));
  const auto &[SuccNode, EdgeTag] = Edge;

  auto InnerSize = SuccNode->Size;
  revng_assert(InnerSize);

  const OffsetExpression &OE = EdgeTag->getOffsetExpr();
  revng_assert(OE.TripCounts.size() == OE.Strides.size());

  // In OE, by construction, strides go to from larger to smaller, so we need to
  // iterate in reverse to build check the strides bottom-up
  for (const auto &[TripCount, Stride] :
       llvm::reverse(llvm::zip(OE.TripCounts, OE.Strides))) {

    // If the Stride is smaller than the inner size the strided edge is not
    // iterating over an entire element, so it's not well formed.
    revng_assert(Stride > 0LL);
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

  if (Log.isEnabled())
    TS.dumpDotOnFile("before-remove-invalid-stride-edges.dot");
  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG() and TS.verifyInheritanceTree());

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
      for (; It != End; It = Next) {

        Next = std::next(It);

        if (not isInstanceEdge(*It))
          continue;

        if (hasValidStrides(*It))
          continue;

        LayoutTypeSystemNode *Succ = It->first;
        auto PredIt = Succ->Predecessors.find({ N, It->second });
        revng_assert(PredIt != Succ->Predecessors.end());
        Succ->Predecessors.erase(PredIt);
        N->Successors.erase(It);
      }
    }
  }

  if (Log.isEnabled())
    TS.dumpDotOnFile("after-remove-invalid-stride-edges.dot");
  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG() and TS.verifyInheritanceTree());

  return Changed;
}

} // end namespace dla
