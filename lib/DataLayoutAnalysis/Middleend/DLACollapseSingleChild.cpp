//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"

#include "DLAStep.h"

using LTSN = dla::LayoutTypeSystemNode;
using PointerFilterT = EdgeFilteredGraph<LTSN *, dla::isPointerEdge>;

using namespace llvm;

static Logger<> Log("dla-collapse-single-child");

namespace dla {
bool CollapseSingleChild::collapseSingle(LayoutTypeSystem &TS,
                                         LayoutTypeSystemNode *Node) {
  LoggerIndent Indent{ Log };
  bool Changed = false;

  auto HasSingleNonStridedChild = [](const LTSN *Node) {
    const auto &SuccBegin = Node->Successors.begin();
    return (Node->Successors.size() == 1) and isInstanceEdge(*SuccBegin)
           and SuccBegin->second->getOffsetExpr().Strides.empty();
  };

  // Get nodes that have a single instance-at-offset-0 child
  bool Merged = true;
  while (Merged and HasSingleNonStridedChild(Node)) {
    auto &ChildEdge = *(Node->Successors.begin());
    auto &Off = ChildEdge.second->getOffsetExpr().Offset;
    LTSN *Child = ChildEdge.first;

    Merged = false;

    if (not Off) {
      // If the only child is at offset 0, and it has the same size as the
      // parent node, the two nodes are indistinguishable, hence they can be
      // merged.

      revng_log(Log, "Has single child at offset 0. Child: " << Child->ID);
      LoggerIndent MoreIndent{ Log };

      // If the parent has a size different from the child, bail out.
      const auto ChildSize = Child->Size;
      if (ChildSize != Node->Size) {
        revng_log(Log,
                  "Size mismatch! Node = " << Node->Size
                                           << " Child = " << ChildSize);
        break;
      }

      revng_log(Log, "Collapsing " << Child->ID << " into " << Node->ID);

      TS.mergeNodes({ /*Into=*/Node, /*From=*/Child });

      Changed = true;
      Merged = true;

    } else if (isPointerRoot(Node) and not isInstanceRoot(Node)) {
      // If the node doesn't have incoming pointer nodes, we can try to absorbe
      // even instance edges with offset different than zero, pushing the
      // padding out of the parent Node.
      // If it has an incoming pointer node we cannot do anything because we'd
      // change the size of the pointee (i.e. the Child)

      // If the parent has a size different from the child + offset, bail out.
      const auto ChildSize = Child->Size;
      if ((ChildSize + Off) != Node->Size) {
        revng_log(Log,
                  "Size mismatch! Node = " << Node->Size << " Child = "
                                           << ChildSize << " Offset = " << Off);
        break;
      }

      // If we would end up merging a NonScalar node into the the non-NonScalar,
      // we'd end up losing information about the size of the NonScalar one,
      // which is wrong since it's fixed and it comes from the Model. In that
      // case we bail out.
      if (Child->NonScalar) {
        revng_log(Log,
                  "Cannot merge NonScalar Child into Node! Node = "
                    << Node->Size << " Child = " << ChildSize
                    << " Offset = " << Off);
        break;
      }

      // Move Node's predecessor edges to Child, adding Off.
      auto PredIt = Node->Predecessors.begin();
      auto PredEnd = Node->Predecessors.end();
      while (PredIt != PredEnd) {
        auto Next = std::next(PredIt);
        TS.moveEdgeTarget(Node, Child, PredIt, Off);
        PredIt = Next;
      }

      TS.mergeNodes({ /*Into=*/Node, /*From=*/Child });
      Node->Size = ChildSize;

      Changed = true;
      Merged = true;
    }
  }

  return Changed;
}

bool CollapseSingleChild::runOnTypeSystem(LayoutTypeSystem &TS) {
  bool Changed = false;
  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());

  for (LTSN *Node : llvm::nodes(&TS)) {
    revng_log(Log, "Analyzing Node: " << Node->ID);
    Changed |= collapseSingle(TS, Node);
  }

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());

  return Changed;
}

} // end namespace dla
