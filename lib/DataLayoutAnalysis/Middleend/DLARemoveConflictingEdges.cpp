//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include <iterator>

#include "llvm/ADT/STLExtras.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"

#include "../DLAHelpers.h"
#include "DLAStep.h"

using LTSN = dla::LayoutTypeSystemNode;
using VecToCollapseT = std::vector<LTSN *>;
using Link = dla::LayoutTypeSystemNode::Link;

using namespace llvm;

static Logger<> Log("dla-remove-conflicting-edges");

namespace dla {

/// \brief Drop instance-at-offset-0 edges when they connect two nodes that
/// are also connected by an inheritance edge.
bool RemoveConflictingEdges::removeConflicts(LayoutTypeSystem &TS,
                                             LayoutTypeSystemNode *Node) {
  bool Changed = false;
  llvm::SmallPtrSet<LTSN *, 8> InhNodes;
  for (auto &L : Node->Successors)
    if (isInheritanceEdge(L))
      InhNodes.insert(L.first);

  auto It = Node->Successors.begin();
  while (It != Node->Successors.end()) {
    auto &L = *It;
    if (isInstanceOff0Edge(L) and InhNodes.contains(L.first)) {
      // Remove from successor's predecessors
      size_t NFound = std::erase_if(L.first->Predecessors,
                                    [Node](const Link &Pred) {
                                      return Pred.first->ID == Node->ID
                                             and isInstanceOff0Edge(Pred);
                                    });
      revng_assert(NFound > 0);

      It = Node->Successors.erase(It);
      Changed = true;
    } else {
      ++It;
    }
  }

  return Changed;
}

bool RemoveConflictingEdges::runOnTypeSystem(LayoutTypeSystem &TS) {
  bool Changed = false;
  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG() and TS.verifyInheritanceTree());
  if (Log.isEnabled())
    TS.dumpDotOnFile("before-remove-conflicting-edges.dot");

  for (LTSN *Node : llvm::nodes(&TS))
    Changed |= removeConflicts(TS, Node);

  if (Log.isEnabled())
    TS.dumpDotOnFile("after-remove-conflicting-edges.dot");
  if (VerifyLog.isEnabled()) {
    revng_assert(TS.verifyInheritanceDAG());
    revng_assert(TS.verifyInheritanceTree());
  }

  return Changed;
}
} // namespace dla