//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"

#include "revng/DataLayoutAnalysis/DLATypeSystem.h"

#include "DLAStep.h"

using namespace llvm;

static Logger<> Log("dla-merge-pointer-nodes");

namespace dla {

using LTSN = LayoutTypeSystemNode;
using GraphNodeT = LTSN *;
using PointerGraphNodeT = EdgeFilteredGraph<GraphNodeT, isPointerEdge>;
using InversePointerGraphNodeT = llvm::Inverse<PointerGraphNodeT>;

bool MergePointerNodes::runOnTypeSystem(LayoutTypeSystem &TS) {
  bool Changed = false;

  for (LTSN *Node : llvm::nodes(&TS)) {
    revng_assert(Node != nullptr);

    revng_log(Log, "# Starting from Node: " << Node->ID);
    for (auto *Node : llvm::depth_first(InversePointerGraphNodeT(Node))) {
      if (isPointerRoot(Node))
        continue;

      std::vector<LTSN *> ToMerge;

      // Look at the pointers to Node. The all have to be merged together.
      // Doing so does not invalidate the depth_first iterator we're holding,
      // because that points to Node, and the iterator only looks at
      // predecessors of Node when it's incremented. So until then we can
      // change the predecessors of Node without problems.
      for (LTSN *PointerToMerge :
           llvm::children<InversePointerGraphNodeT>(Node)) {
        revng_log(Log, "#### PointerToMerge: " << PointerToMerge->ID);
        revng_assert(PointerToMerge != Node);
        revng_assert(PointerToMerge->Successors.size() == 1);
        revng_assert(ToMerge.empty()
                     or ToMerge[0]->Size == PointerToMerge->Size);
        ToMerge.push_back(PointerToMerge);
      }

      if (ToMerge.size() < 2)
        continue;

      Changed = true;

      // This does note invalidate our iteration on llvm::nodes, since the
      // underlying container for nodes is a std::set, and the iterator that is
      // being held on it points to Node, not to any of the nodes in ToMerge,
      // so erasing the nodes in ToMerge never invalidats the iterator to Node
      TS.mergeNodes(ToMerge);
    }
  }

  return Changed;
}

} // end namespace dla
