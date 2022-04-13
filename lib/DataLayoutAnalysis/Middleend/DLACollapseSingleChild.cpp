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
using VecToCollapseT = std::vector<LTSN *>;
using NonPointerFilterT = EdgeFilteredGraph<LTSN *, dla::isNotPointerEdge>;
using Link = dla::LayoutTypeSystemNode::Link;

using namespace llvm;

static Logger<> Log("dla-collapse-single-child");

namespace dla {
bool CollapseSingleChild::collapseSingle(LayoutTypeSystem &TS,
                                         LayoutTypeSystemNode *Node) {
  bool Changed = false;

  auto HasSingleChild = [](const LTSN *Node) {
    return (Node->Successors.size() == 1);
  };

  auto HasAtMostOneParent = [](const LTSN *Node) {
    return (Node->Predecessors.size() <= 1);
  };

  if (VerifyLog.isEnabled() and not HasSingleChild(Node)) {
    revng_assert(llvm::none_of(Node->Successors,
                               [](const Link &L) { return isPointerEdge(L); }));
  }

  // Get nodes that have a single instance or inheritance child
  if (HasSingleChild(Node) and isInstanceEdge(*Node->Successors.begin())) {
    auto &ChildEdge = *(Node->Successors.begin());
    const unsigned ChildOffset = ChildEdge.second->getOffsetExpr().Offset;
    auto &ToMerge = ChildEdge.first;

    // Don't collapse if the child has more than one parent
    if (not HasAtMostOneParent(ToMerge))
      return false;

    // Collapse only if the child is at offset 0
    if (ChildOffset == 0) {
      revng_log(Log, "Collapsing " << ToMerge->ID << " into " << Node->ID);

      const unsigned ChildSize = ToMerge->Size;
      revng_assert(Node->Size == 0 or Node->Size >= ChildSize);

      // Merge single child into parent.
      // mergeNodes resets InterferingInfo of Node, but we're collapsing
      // ToMerge that is the only child of Node, so we have to attach ToMerge's
      // InterferingInfo to the parent Node that we're preserving.
      // This is always correct because the InterferingInfo of a node only
      // depend on its children.
      auto ChildInterferingInfo = ToMerge->InterferingInfo;
      TS.mergeNodes({ /*Into=*/Node, /*From=*/ToMerge });
      Node->Size = ChildSize;
      Node->InterferingInfo = ChildInterferingInfo;

      Changed = true;
    }
  }

  return Changed;
}

bool CollapseSingleChild::runOnTypeSystem(LayoutTypeSystem &TS) {
  bool Changed = false;
  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());

  for (LTSN *Root : llvm::nodes(&TS)) {
    revng_assert(Root != nullptr);
    if (not isRoot(Root))
      continue;

    for (LTSN *Node : post_order(Root))
      Changed |= collapseSingle(TS, Node);
  }

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());

  return Changed;
}

} // end namespace dla
