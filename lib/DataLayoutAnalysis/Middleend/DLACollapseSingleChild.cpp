//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
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

  auto ChildIsInstanceOrInheritance = [](const LTSN *Node) {
    auto &Child = *Node->Successors.begin();
    return (isInstanceEdge(Child) or isInheritanceEdge(Child));
  };

  auto HasAtMostOneParent = [](const LTSN *Node) {
    return (Node->Predecessors.size() <= 1);
  };

  if (VerifyLog.isEnabled() and not HasSingleChild(Node)) {
    revng_assert(llvm::none_of(Node->Successors,
                               [](const Link &L) { return isPointerEdge(L); }));
  }

  // Get nodes that have a single instance or inheritance child
  if (HasSingleChild(Node) and ChildIsInstanceOrInheritance(Node)) {
    auto &ChildEdge = *(Node->Successors.begin());
    const unsigned ChildOffset = isInheritanceEdge(ChildEdge) ?
                                   0U :
                                   ChildEdge.second->getOffsetExpr().Offset;
    auto &ToMerge = ChildEdge.first;

    // Don't collapse if the child has more than one parent
    if (not HasAtMostOneParent(ToMerge))
      return false;

    // Collapse only if the child is at offset 0
    if (ChildOffset == 0) {
      revng_log(Log, "Collapsing " << ToMerge->ID << " into " << Node->ID);

      const unsigned ChildSize = ToMerge->Size;
      revng_assert(Node->Size == 0 or Node->Size >= ChildSize);

      // Merge single child into parent
      TS.mergeNodes({ /*Into=*/Node, /*From=*/ToMerge });
      Node->Size = ChildSize;

      Changed = true;
    }
  }

  return Changed;
}

bool CollapseSingleChild::runOnTypeSystem(LayoutTypeSystem &TS) {
  bool Changed = false;
  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG() and TS.verifyInheritanceTree());

  if (Log.isEnabled())
    TS.dumpDotOnFile("before-collapse-single-child.dot");

  for (LTSN *Root : llvm::nodes(&TS)) {
    revng_assert(Root != nullptr);
    if (not isRoot(Root))
      continue;

    for (LTSN *Node : post_order(Root))
      Changed |= collapseSingle(TS, Node);
  }

  if (Log.isEnabled())
    TS.dumpDotOnFile("after-collapse-single-child.dot");
  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyInheritanceDAG() and TS.verifyInheritanceTree());

  return Changed;
}

} // end namespace dla
