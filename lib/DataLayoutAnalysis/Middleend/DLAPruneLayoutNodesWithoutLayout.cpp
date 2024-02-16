//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>
#include <set>

#include "llvm/ADT/PostOrderIterator.h"

#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/Support/Debug.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"

#include "DLAStep.h"

using namespace llvm;

static Logger<> Log("dla-prune");

namespace dla {

using LTSN = LayoutTypeSystemNode;
using GraphNodeT = LTSN *;
using NonPointerFilterT = EdgeFilteredGraph<GraphNodeT, isNotPointerEdge>;

bool PruneLayoutNodesWithoutLayout::runOnTypeSystem(LayoutTypeSystem &TS) {
  bool Changed = false;

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());

  std::set<const LTSN *> Visited;
  std::set<LTSN *> ToRemove;

  for (LTSN *Root : llvm::nodes(&TS)) {
    revng_assert(Root != nullptr);
    if (not isRoot(Root))
      continue;

    revng_log(Log, "# Starting from Root: " << Root->ID);
    for (LTSN *N : post_order_ext(NonPointerFilterT(Root), Visited)) {
      revng_log(Log, "## Visiting N: " << N->ID);
      revng_log(Log, "## Is Leaf: " << isLeaf(N));

      if (N->Size > 0) {
        revng_log(Log, "### has size " << N->Size << " !");
        continue;
      }

      using GT = GraphTraits<NonPointerFilterT>;
      if (std::any_of(GT::child_begin(N),
                      GT::child_end(N),
                      [&ToRemove](LTSN *Child) {
                        return !ToRemove.contains(Child);
                      })) {
        revng_log(Log, "### ChildHasLayout(N)!");
        continue;
      }

      // Here N does not have valid layout, nor any of its child has.
      revng_log(Log, "#### Queuing for removal: " << N->ID);
      ToRemove.insert(N);
    }
  }

  Changed = not ToRemove.empty();

  for (LTSN *N : ToRemove) {
    revng_log(Log, "# Removing: " << N->ID);
    TS.removeNode(N);
  }

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG() and TS.verifyLeafs());

  return Changed;
}

} // end namespace dla
