//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <iostream>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include "revng/Support/Debug.h"

#include "DLAStep.h"

using InstanceT = EdgeFilteredGraph<dla::LayoutTypeSystemNode *,
                                    dla::isInstanceEdge>;

using InstanceZeroT = EdgeFilteredGraph<dla::LayoutTypeSystemNode *,
                                        dla::isInstanceOff0>;

using CInstanceT = EdgeFilteredGraph<const dla::LayoutTypeSystemNode *,
                                     dla::isInstanceEdge>;

using InverseCInstanceT = llvm::Inverse<CInstanceT>;

using InverseInstanceT = llvm::Inverse<InstanceT>;

using CPointerT = EdgeFilteredGraph<const dla::LayoutTypeSystemNode *,
                                    dla::isPointerEdge>;

static Logger<> Log{ "dla-simplify-instance-off0" };

namespace dla {

using NodeSet = llvm::SmallPtrSet<const LayoutTypeSystemNode *, 8>;
using ReachabilityMap = llvm::DenseMap<const LayoutTypeSystemNode *, NodeSet>;

static bool hasOutgoingPointer(const LayoutTypeSystemNode *N) {
  using PointerGraph = llvm::GraphTraits<CPointerT>;
  auto It = PointerGraph::child_begin(N);
  auto End = PointerGraph::child_end(N);
  return It != End;
};

class ReachabilityCache {
  NodeSet Visited;
  ReachabilityMap CanReach;

public:
  bool existsPath(const LayoutTypeSystemNode *From,
                  const LayoutTypeSystemNode *To) {
    revng_log(Log, "existPath: " << From->ID << " -> " << To->ID);
    LoggerIndent ExistsIndent(Log);

    if (auto It = CanReach.find(From); It != CanReach.end()) {
      bool Exists = It->second.contains(To);
      revng_log(Log, "cached: " << (Exists ? 'Y' : 'N'));
      return Exists;
    }

    {
      revng_log(Log, "Compute reachability for: " << From->ID);
      LoggerIndent ReachabilityIndent(Log);

      for (const LayoutTypeSystemNode *Node :
           llvm::post_order_ext(CInstanceT(From), Visited)) {
        NodeSet &ReachableFromNode = CanReach[Node];
        revng_log(Log, "Node: " << From->ID);
        LoggerIndent NodeIndent(Log);

        for (const LayoutTypeSystemNode *Child :
             llvm::children<CInstanceT>(Node)) {

          revng_log(Log, "can reach child: " << Child->ID);
          LoggerIndent ChildIndent(Log);

          ReachableFromNode.insert(Child);

          auto ReachableFromChildIt = CanReach.find(Child);
          revng_assert(ReachableFromChildIt != CanReach.end());
          const NodeSet &ReachableFromChild = ReachableFromChildIt->second;

          for (const LayoutTypeSystemNode *R : ReachableFromChild) {
            revng_log(Log, "can reach: " << R->ID << " transitive");
            ReachableFromNode.insert(R);
          }
        }
      }
    }

    auto ReachableIt = CanReach.find(From);
    revng_assert(ReachableIt != CanReach.end());
    const NodeSet &Reachable = ReachableIt->second;
    bool Exists = Reachable.contains(To);
    revng_log(Log, "computed: " << (Exists ? 'Y' : 'N'));
    return Exists;
  }
};

static bool canBeCollapsed(ReachabilityCache Cache,
                           const LayoutTypeSystemNode *Node,
                           const LayoutTypeSystemNode *Child) {
  LoggerIndent Indent(Log);

  // If Child has Size and the size is different from its parent Node, it cannot
  // be collapsed.
  if (Child->Size and Child->Size != Node->Size) {
    revng_log(Log, "has Size");
    return false;
  }

  // If Child has an outgoing pointer it cannot be collapsed.
  if (hasOutgoingPointer(Child)) {
    revng_log(Log, "has outgoing pointer");
    return false;
  }

  // If the parent has other children, Child cannot be collapsed in it, because
  // that would mean that all other types that contain an instance of Child
  // would start also containing instances of subtypes of Node.
  // The only case where it is allowed is when the only predecessor of Child is
  // Node, because in that case we there is no risk of other predecessors of
  // Child to see other nodes that were originally other children of Node.
  if (Node->Successors.size() > 1 and Child->Predecessors.size() > 1) {
    revng_log(Log, "parent has other children");
    return false;
  }

  revng_assert(Node->Successors.size() > 1
               or (isInstanceOff0(*Node->Successors.begin())
                   and Node->Successors.begin()->first == Child));

  // In all the other cases we can collapse.
  return true;
}

static LayoutTypeSystemNode *addArtificialRoot(LayoutTypeSystem &TS) {

  LayoutTypeSystemNode *FakeRoot = TS.createArtificialLayoutType();
  revng_log(Log, "Adding FakeRoot ID: " << FakeRoot->ID);
  LoggerIndent Indent(Log);
  for (LayoutTypeSystemNode *Node : llvm::nodes(&TS)) {
    if (Node == FakeRoot)
      continue;

    if (isInstanceRoot(Node)) {
      revng_log(Log,
                "adding Instance-0 edge: " << FakeRoot->ID << " -> "
                                           << Node->ID);
      TS.addInstanceLink(FakeRoot, Node, OffsetExpression{});
    }
  }
  return FakeRoot;
}

static llvm::SmallVector<LayoutTypeSystemNode *>
getPostOrder(LayoutTypeSystemNode *Root) {

  llvm::SmallVector<LayoutTypeSystemNode *> PostOrder;
  revng_log(Log, "PostOrder:");
  LoggerIndent Indent(Log);
  for (LayoutTypeSystemNode *Node : llvm::post_order(InstanceT(Root))) {
    revng_log(Log, Node->ID);
    PostOrder.push_back(Node);
  }
  return PostOrder;
}

bool SimplifyInstanceAtOffset0::runOnTypeSystem(LayoutTypeSystem &TS) {

  if (Log.isEnabled())
    TS.dumpDotOnFile("before-SimplifyInstanceAtOffset0.dot", true);

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyInstanceDAG());

  LayoutTypeSystemNode *FakeRoot = addArtificialRoot(TS);
  llvm::SmallVector<LayoutTypeSystemNode *> PostOrder = getPostOrder(FakeRoot);

  bool Changed = false;

  uint64_t I = 0;

  revng_log(Log, "Process Nodes:");
  LoggerIndent Indent(Log);
  for (LayoutTypeSystemNode *Node : PostOrder) {

    if (Node == FakeRoot)
      continue;

    revng_log(Log, "Node: " << Node->ID);
    LoggerIndent NodeIndent(Log);

    ReachabilityCache RC;

    // For each instance children of Node at offset 0, compute if it can be
    // collapsed into Node, and if it's possible collapse it.
    for (LayoutTypeSystemNode *Child :
         llvm::make_early_inc_range(llvm::children<InstanceZeroT>(Node))) {

      revng_log(Log, "Child: " << Child->ID);
      LoggerIndent ChildIndent(Log);

      if (not canBeCollapsed(RC, Node, Child)) {
        revng_log(Log, "NOT canBeCollapsed()");
        continue;
      }

      revng_log(Log, "canBeCollapsed()");

      auto IDToCollapse = Child->ID;
      if (Log.isEnabled()) {
        TS.dumpDotOnFile((llvm::Twine(I) + "-before-"
                          + llvm::Twine(IDToCollapse) + ".dot")
                           .str(),
                         true);
      }

      TS.mergeNodes({ Node, Child });
      Changed = true;

      if (Log.isEnabled()) {
        TS.dumpDotOnFile((llvm::Twine(I) + "-after-" + llvm::Twine(IDToCollapse)
                          + ".dot")
                           .str(),
                         true);
        ++I;
        revng_assert(TS.verifyInstanceDAG());
      }
    }
  }

  TS.removeNode(FakeRoot);

  if (Log.isEnabled())
    TS.dumpDotOnFile("after-SimplifyInstanceAtOffset0.dot", true);

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyInstanceDAG());

  return Changed;
}

} // end namespace dla
