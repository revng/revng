//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/PostOrderIterator.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"

#include "../DLAHelpers.h"
#include "DLAStep.h"

using LTSN = dla::LayoutTypeSystemNode;
using VecToCollapseT = std::vector<LTSN *>;

using namespace llvm;

static Logger<> Log("dla-collapse-single-child");

namespace dla {

bool CollapseSingleChild::runOnTypeSystem(LayoutTypeSystem &TS) {
  bool Changed = false;
  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());

  if (Log.isEnabled())
    TS.dumpDotOnFile("before-collapse-single-child.dot");

  auto HasSingleChild = [](const LTSN *Node) {
    return (Node->Successors.size() == 1);
  };

  auto FirstChildIsInstance = [](const LTSN *Node) {
    return (isInstanceEdge(*Node->Successors.begin()));
  };

  // Find roots
  for (LTSN *Root : llvm::nodes(&TS)) {
    revng_assert(Root != nullptr);
    if (not isRoot(Root))
      continue;

    // Visit their sub-tree in post order
    for (LTSN *Node : post_order(Root)) {
      if (HasSingleChild(Node) and FirstChildIsInstance(Node)) {
        auto &ChildEdge = *(Node->Successors.begin());
        auto &OE = ChildEdge.second->getOffsetExpr();
        auto &ToMerge = ChildEdge.first;

        // Get nodes that have a single instance child at offset 0
        if (OE.Offset == 0) {
          revng_log(Log, "Collapsing " << ToMerge->ID << " into " << Node->ID);

          const unsigned ChildSize = ToMerge->Size;
          revng_assert(Node->Size == 0);

          // Merge single child into parent
          TS.mergeNodes({ /*Into=*/Node, /*From=*/ToMerge });
          Node->InterferingInfo = AllChildrenAreNonInterfering;
          Node->Size = ChildSize;

          Changed = true;
        }
      }
    }
  }

  if (Log.isEnabled())
    TS.dumpDotOnFile("after-collapse-single-child.dot");
  if (VerifyLog.isEnabled()) {
    revng_assert(TS.verifyInheritanceDAG());
    revng_assert(TS.verifyInheritanceTree());
  }

  return Changed;
}

} // end namespace dla
