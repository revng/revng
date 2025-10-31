//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>
#include <type_traits>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Debug.h"

#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/DataLayoutAnalysis/DLATypeSystem.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#include "DLAStep.h"
#include "FieldSizeComputation.h"

using namespace llvm;

static Logger Log("dla-compute-upper-member-access");

namespace dla {

using LTSN = LayoutTypeSystemNode;
using GraphNodeT = LTSN *;
using NonPointerFilterT = EdgeFilteredGraph<GraphNodeT, isNotPointerEdge>;
using ConstNonPointerFilterT = EdgeFilteredGraph<const LTSN *,
                                                 isNotPointerEdge>;

bool ComputeUpperMemberAccesses::runOnTypeSystem(LayoutTypeSystem &TS) {
  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());
  bool Changed = false;

  using LTSN = LayoutTypeSystemNode;
  std::set<const LTSN *> Visited;
  for (LTSN *Root : llvm::nodes(&TS)) {
    revng_log(Log, "Root ID: " << Root->ID);
    revng_assert(Root != nullptr);
    // Leaves need to have ValidLayouts, otherwise they should have been trimmed
    // by PruneLayoutNodesWithoutLayout
    revng_assert(not isLeaf(Root) or Root->Size);
    if (not isRoot(Root))
      continue;

    LoggerIndent Indent{ Log };
    revng_log(Log, "Is Root");
    revng_log(Log, "post_order_ext from Root");
    LoggerIndent MoreIndent{ Log };

    for (LTSN *N : post_order_ext(NonPointerFilterT(Root), Visited)) {
      revng_log(Log, "N->ID: " << N->ID);
      revng_assert(not isLeaf(N) or N->Size);
      uint64_t FinalSize = N->Size;

      // Look at all the instance-of edges and inheritance edges all together.
      revng_log(Log, "N's children");
      LoggerIndent MoreMoreIndent{ Log };
      for (auto &[Child, EdgeTag] : children_edges<ConstNonPointerFilterT>(N)) {
        revng_log(Log, "Child->ID: " << Child->ID);
        revng_log(Log,
                  "EdgeTag->Kind: "
                    << dla::TypeLinkTag::toString(EdgeTag->getKind()));
        FinalSize = std::max(FinalSize, getFieldUpperMember(Child, EdgeTag));
      }

      if (FinalSize != N->Size)
        Changed = true;

      N->Size = FinalSize;
      revng_assert(FinalSize);
    }
  }

  return Changed;
}

} // end namespace dla
