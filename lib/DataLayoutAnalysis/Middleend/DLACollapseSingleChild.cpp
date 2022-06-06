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
  LoggerIndent Indent{ Log };
  bool Changed = false;

  auto HasSingleChildAtOffset0 = [](const LTSN *Node) {
    return (Node->Successors.size() == 1)
           and isInstanceOff0(*Node->Successors.begin());
  };

  // Get nodes that have a single instance-at-offset-0 child
  while (HasSingleChildAtOffset0(Node)) {
    auto &ChildEdge = *(Node->Successors.begin());
    LTSN *Child = ChildEdge.first;

    revng_log(Log, "Has single child at offset 0. Child: " << Child->ID);
    LoggerIndent MoreIndent{ Log };

    // If the parent has a size different from the child, bail out.
    const unsigned ChildSize = Child->Size;
    if (ChildSize != Node->Size) {
      revng_log(Log,
                "Size mismatch! Node = " << Node->Size
                                         << " Child = " << ChildSize);
      break;
    }

    revng_log(Log, "Collapsing " << Child->ID << " into " << Node->ID);

    // Merge single child into parent.
    // mergeNodes resets InterferingInfo of Node, but we're collapsing
    // Child that is the only child of Node, so we have to attach Child's
    // InterferingInfo to the parent Node that we're preserving.
    // This is always correct because the InterferingInfo of a node only
    // depend on its children.
    auto ChildInterferingInfo = Child->InterferingInfo;
    TS.mergeNodes({ /*Into=*/Node, /*From=*/Child });
    Node->InterferingInfo = ChildInterferingInfo;

    Changed = true;
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
