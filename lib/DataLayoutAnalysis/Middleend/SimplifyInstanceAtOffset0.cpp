//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
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
  // If Child has size or has an outgoing pointer edge, it can never be
  // collapsed.
  if (Child->Size) {
    revng_log(Log, "has Size");
    return false;
  }

  if (hasOutgoingPointer(Child)) {
    revng_log(Log, "has outgoing pointer");
    return false;
  }

  // If we find a successor of Node (different from Child at offset 0) such that
  // it can reach Child, then the instance-0 from Node to Child cannot be
  // collapsed, because that would introduce a loop.

  // First check the cheapest thing, i.e. if there is another edge to Child that
  // is not at offset 0. That is cheap to check and would already mean that we
  // cannot collapse, because collapsing the instance-0 edge would turn the
  // other edge to Child into a self-loop.
  for (const auto &Link : llvm::children_edges<CInstanceT>(Node)) {
    const auto &[OtherChild, EdgeTag] = Link;
    if (Child == OtherChild and not isInstanceOff0(Link)) {
      revng_log(Log, "has another incoming edge");
      return false;
    }
  }

  // Then check if Child is actually reachable from any other instance children
  // (OtherChildren) of Node.
  // If that happens, collapsing Child into Node, would create a loop, so we
  // have to bail out.
  revng_log(Log, "check mutual reachability");
  LoggerIndent MoreIndent(Log);
  for (const LayoutTypeSystemNode *OtherChild :
       llvm::children<CInstanceT>(Node)) {
    // Ignore the edge that would be collapsed.
    if (OtherChild == Child)
      continue;

    revng_log(Log, "OtherChild: " << OtherChild->ID);
    LoggerIndent MoreMoreIndent(Log);

    // We're using the cache here since otherwise we might end up exploring a
    // big chunk of the graph very many times, as experiments on real-world
    // examples have shown.
    if (Cache.existsPath(OtherChild, Child)) {
      revng_log(Log, "can reach: " << Child->ID);
      return false;
    }
    revng_log(Log, "cannot reach: " << Child->ID << " BUT: ");
  }

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
