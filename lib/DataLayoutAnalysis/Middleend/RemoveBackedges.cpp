//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>
#include <limits>
#include <vector>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Progress.h"

#include "revng/DataLayoutAnalysis/DLATypeSystem.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#include "RemoveBackedges.h"

static Logger<> Log("dla-remove-backedges");

namespace dla {

using LTSN = LayoutTypeSystemNode;

template<typename T>
concept SCCWithBackedgeHelper = requires {
  typename T::SCCNodeView;
  typename T::BackedgeNodeView;
};

struct InstanceOffsetZeroWithInstanceBackedge {
  using SCCNodeView = EdgeFilteredGraph<const LTSN *, isInstanceOff0>;
  using BackedgeNodeView = EdgeFilteredGraph<const LTSN *, isInstanceOffNon0>;
};

template<SCCWithBackedgeHelper SCC>
static bool isMixedEdge(const llvm::GraphTraits<dla::LTSN *>::EdgeRef &E) {
  return SCC::SCCNodeView::filter()(E) or SCC::BackedgeNodeView::filter()(E);
}

template<SCCWithBackedgeHelper SCC>
static bool isSCCLeaf(const dla::LTSN *N) {
  using Graph = llvm::GraphTraits<typename SCC::SCCNodeView>;
  return Graph::child_begin(N) == Graph::child_end(N);
}

template<SCCWithBackedgeHelper SCC>
static bool isSCCRoot(const dla::LTSN *N) {
  using Inverse = llvm::Inverse<typename SCC::SCCNodeView>;
  using InverseGraph = llvm::GraphTraits<Inverse>;
  return InverseGraph::child_begin(N) == InverseGraph::child_end(N);
}

template<SCCWithBackedgeHelper SCC>
static bool hasNoSCCEdge(const LTSN *Node) {
  return isSCCRoot<SCC>(Node) and isSCCLeaf<SCC>(Node);
};

template<SCCWithBackedgeHelper SCC>
using MixedNodeT = EdgeFilteredGraph<LTSN *, isMixedEdge<SCC>>;

template<SCCWithBackedgeHelper SCC>
struct StackEntry {
  LTSN *Node = nullptr;
  const LTSN *ComponentLeader = nullptr;
  typename MixedNodeT<SCC>::ChildEdgeIteratorType NextToVisitIt;
};

struct EdgeInfo {
  LTSN *Src = nullptr;
  LTSN *Tgt = nullptr;
  const TypeLinkTag *Tag = nullptr;
  // Comparison operators to use in set
  std::strong_ordering operator<=>(const EdgeInfo &) const = default;
};

template<SCCWithBackedgeHelper SCC>
static bool removeBackedgesFromSCC(LayoutTypeSystem &TS) {
  bool Changed = false;
  if (VerifyLog.isEnabled()) {
    revng_assert(TS.verifyConsistency());

    // Verify that the graph is a DAG looking only at SCCNodeView
    std::set<const LTSN *> Visited;
    for (const auto &Node : llvm::nodes(&TS)) {
      revng_assert(Node != nullptr);
      if (Visited.contains(Node))
        continue;

      auto I = llvm::scc_begin(typename SCC::SCCNodeView(Node));
      auto E = llvm::scc_end(typename SCC::SCCNodeView(Node));
      for (; I != E; ++I) {
        Visited.insert(I->begin(), I->end());
        if (I.hasCycle())
          revng_check(false);
      }
    }
  }

  revng_log(Log, "Removing Backedges From Loops");

  llvm::Task T(2, "removeBackedgesFromSCC");
  T.advance("Detect SCC Node View Components");
  // Assign each node to a Component, except for those that have no incoming nor
  // outgoing SCCNodeView edges. The goal is to identify the subsets of nodes
  // that are connected by means of SCCNodeView edges. In this way we divide the
  // graph in subgraphs, such that for each pair of nodes P and Q with (P != Q)
  // in the same sugraphs (i.e. with the same component) P is reachable from Q
  // looking only at **undirected** SCCNodeView edges. Each of this subgraphs is
  // called "component". The idea is that SCCNodeView edges are more meaningful
  // than SCCBackedgeView edges, so we don't want to remove any of them, but we
  // need to identify such edges that create loops across multiple components,
  // and cut them.
  llvm::EquivalenceClasses<const LTSN *> Components;
  {
    revng_log(Log, "Detect components");
    for (const LTSN *N : llvm::nodes(&TS)) {
      revng_assert(N != nullptr);
      revng_log(Log, "N->ID: " << N->ID);
      LoggerIndent Indent{ Log };

      for (const LTSN *Child : llvm::children<typename SCC::SCCNodeView>(N)) {
        revng_log(Log, "Merging with SCCNodeView Child with ID: " << Child->ID);
        Components.unionSets(N, Child);
      }
    }
  }

  if (Log.isEnabled()) {
    revng_log(Log, "Detected components:");
    LoggerIndent Indent{ Log };
    for (auto I = Components.begin(), E = Components.end(); I != E;
         ++I) { // Iterate over all of the equivalence sets.

      if (!I->isLeader()) {
        // Ignore non-leader sets.
        continue;
      }

      revng_log(Log,
                "Component for Node with ID: "
                  << (*Components.findLeader(I))->ID);
      LoggerIndent MoreIndent{ Log };
      // Loop over members in this set.
      for (const LTSN *N : llvm::make_range(Components.member_begin(I),
                                            Components.member_end()))
        revng_log(Log, "ID: " << N->ID);
    }
  }

  // Here all the nodes are nodes have a component, except nodes that have no
  // incoming or outgoing SCCNodeView edges

  T.advance("Remove Backedges");
  using MixedNodeT = MixedNodeT<SCC>;

  for (const auto &Root : llvm::nodes(&TS)) {
    revng_assert(Root != nullptr);
    // We start from SCCNodeView roots and look if we find an SCC with mixed
    // edges.
    if (hasNoSCCEdge<SCC>(Root))
      continue;

    if (not isSCCRoot<SCC>(Root))
      continue;

    revng_log(Log, "# Looking for mixed loops from: " << Root->ID);

    using StackEntry = StackEntry<SCC>;

    llvm::SmallPtrSet<const LTSN *, 16> Visited;
    llvm::SmallPtrSet<const LTSN *, 16> InStack;
    llvm::SmallVector<EdgeInfo> BackedgesOnStack;
    llvm::SmallVector<StackEntry> VisitStack;

    // Helper enum to describe if push succeeds and different ways to fail
    enum class TryPushStatus {
      FailedNoChildren,
      FailedCloseLoop,
      AlreadyVisited,
      Success,
    };

    // Struct holding the success/failure status, along with the edge iterator
    // that caused it.
    struct TryPushResult {
      TryPushStatus Status;
      MixedNodeT::ChildEdgeIteratorType EdgeToPush;
    };

    const auto TryPushNextChild = [&]() -> TryPushResult {
      revng_assert(not VisitStack.empty());
      revng_assert(VisitStack.size() == InStack.size());
      StackEntry &Top = VisitStack.back();
      const auto &[TopNode, TopComponent, _] = Top;
      // Make a copy, because we have to increment the old Top.NextToVisitIt,
      // while at the same time we still want to have access to the previous one
      // in case of failure.
      typename MixedNodeT::ChildEdgeIteratorType
        NextEdgeToVisit = Top.NextToVisitIt;

      revng_log(Log, "--* TryPushNextChild of (" << TopNode->ID << ')');
      LoggerIndent Indent{ Log };

      if (NextEdgeToVisit == MixedNodeT::child_edge_end(TopNode)) {
        revng_log(Log, "--| no children left");
        return TryPushResult{ .Status = TryPushStatus::FailedNoChildren,
                              .EdgeToPush = NextEdgeToVisit };
      }

      ++Top.NextToVisitIt;

      LTSN *NextChild = NextEdgeToVisit->first;
      const TypeLinkTag *NextTag = NextEdgeToVisit->second;
      revng_log(Log,
                "--| next children is " << NextChild->ID
                                        << " tag: " << *NextTag);

      bool NewVisit = Visited.insert(NextChild).second;
      if (not NewVisit) {
        if (InStack.contains(NextChild)) {
          revng_log(Log, "--| loop detected!");
          return TryPushResult{ .Status = TryPushStatus::FailedCloseLoop,
                                .EdgeToPush = NextEdgeToVisit };
        } else {
          revng_log(Log, "--| already visited!");
          return TryPushResult{ .Status = TryPushStatus::AlreadyVisited,
                                .EdgeToPush = NextEdgeToVisit };
        }
      }

      // Make sure we track all the backedges on stack, because if we ever hit
      // a loop we'll have to mark these for removal.
      EdgeInfo NewEdge = { TopNode, NextChild, NextTag };
      if (SCC::BackedgeNodeView::filter()({ nullptr, NewEdge.Tag }))
        BackedgesOnStack.push_back(std::move(NewEdge));

      // Add a new entry on the stack
      revng_log(Log, "TopComponent: " << TopComponent);

      StackEntry NewEntry = {
        .Node = NextChild,
        .ComponentLeader = Components.findValue(NextChild) != Components.end() ?
                             *Components.findLeader(NextChild) :
                             TopComponent,
        .NextToVisitIt = MixedNodeT::child_edge_begin(NextChild)
      };

      VisitStack.push_back(std::move(NewEntry));
      InStack.insert(NextChild);
      revng_assert(InStack.size() == VisitStack.size());

      revng_log(Log, "--> pushed!");
      return TryPushResult{ .Status = TryPushStatus::Success,
                            .EdgeToPush = NextEdgeToVisit };
    };

    const auto Pop = [&]() {
      revng_assert(InStack.size() == VisitStack.size());

      // Make a copy before popping.
      StackEntry Back = VisitStack.back();
      revng_log(Log, "<-- pop(" << Back.Node->ID << ')');

      // Erase the node and the StackEntry from stack
      bool Erased = InStack.erase(Back.Node);
      revng_assert(Erased);
      VisitStack.pop_back();

      // If the stack is now empty, we're done.
      if (VisitStack.empty())
        return;

      // If after popping, the stack isn't empty, we have to check if we've
      // popped a backedge, and in that case remove it from BackedgesOnStack.
      StackEntry &Parent = VisitStack.back();
      auto Begin = MixedNodeT::child_edge_begin(Parent.Node);
      revng_assert(Parent.NextToVisitIt != Begin);

      EdgeInfo PoppedEdge = {
        .Src = Parent.Node,
        .Tgt = Back.Node,
        .Tag = std::prev(Parent.NextToVisitIt)->second,
      };

      // If we've popped a backedge it must be equal to PoppedEdge. If it's not
      // the stack is malformed. If it's the correct edge, we have to remove it
      // from BackedgesOnStack since we're popping.
      if (SCC::BackedgeNodeView::filter()({ nullptr, PoppedEdge.Tag })) {
        revng_assert(BackedgesOnStack.back() == PoppedEdge);
        BackedgesOnStack.pop_back();
      }
    };

    llvm::SmallSet<EdgeInfo, 8> ToRemove;

    revng_assert(Components.findValue(Root) != Components.end());
    StackEntry Init = { .Node = Root,
                        .ComponentLeader = Components.getLeaderValue(Root),
                        .NextToVisitIt = MixedNodeT::child_edge_begin(Root) };
    VisitStack.push_back(std::move(Init));
    InStack.insert(Root);
    Visited.insert(Root);

    while (not VisitStack.empty()) {

      StackEntry &Top = VisitStack.back();

      revng_log(Log, "## Stack top has ID: " << Top.Node->ID);
      revng_log(Log, "   Component ID: " << Top.ComponentLeader->ID);

      const auto &[Status, Edge] = TryPushNextChild();
      switch (Status) {

      default:
        // do nothing
        break;

      case TryPushStatus::FailedNoChildren: {
        Pop();
      } break;

      case TryPushStatus::FailedCloseLoop: {

        // If we're closing a loop, add all the backedges involved in the loop
        // in the edges ToRemove.
        if (Log.isEnabled()) {
          for (EdgeInfo &B : BackedgesOnStack) {
            revng_log(Log,
                      "remove backedge that is on stack: "
                        << B.Src->ID << " -> " << B.Tgt->ID
                        << " Tag: " << B.Tag);
          }
        }
        ToRemove.insert(BackedgesOnStack.begin(), BackedgesOnStack.end());

        if (SCC::BackedgeNodeView::filter()({ nullptr, Edge->second })) {
          // This means that the edge E we tried to push on the stack is an
          // BackedgeNodeView edge closing a loop.
          EdgeInfo BackedgeNotPushed{
            .Src = VisitStack.back().Node,
            .Tgt = Edge->first,
            .Tag = Edge->second,
          };
          revng_log(Log,
                    "remove backedge closing loop, that isn't on stack: "
                      << BackedgeNotPushed.Src->ID << " -> "
                      << BackedgeNotPushed.Tgt->ID
                      << " Tag: " << BackedgeNotPushed.Tag);
          ToRemove.insert(std::move(BackedgeNotPushed));
        }

      } break;
      }
    }

    // Actually remove the edges
    for (auto &[Pred, Child, T] : ToRemove) {
      using Edge = LTSN::NeighborsSet::value_type;
      revng_log(Log,
                "# Removing backedge: " << Pred->ID << " -> " << Child->ID);
      revng_assert(SCC::BackedgeNodeView::filter()({ Pred, T }));
      Edge ChildToPred = std::make_pair(Pred, T);
      bool Erased = Child->Predecessors.erase(ChildToPred);
      revng_assert(Erased);
      Edge PredToChild = std::make_pair(Child, T);
      Erased = Pred->Successors.erase(PredToChild);
      revng_assert(Erased);
      Changed = true;
    }
  }

  if (VerifyLog.isEnabled()) {
    revng_assert(TS.verifyConsistency());

    // Verify that the graph is a DAG looking both at SCCNodeView and
    // BackedgeNodeView
    std::set<const LTSN *> Visited;
    for (const auto &Node : llvm::nodes(&TS)) {
      revng_assert(Node != nullptr);
      if (Visited.contains(Node))
        continue;

      auto I = llvm::scc_begin(MixedNodeT(Node));
      auto E = llvm::scc_end(MixedNodeT(Node));
      for (; I != E; ++I) {
        Visited.insert(I->begin(), I->end());
        if (I.hasCycle()) {
          for (const auto *N : *I)
            llvm::dbgs() << std::to_string(N->ID) << "\n";
          revng_check(false);
        }
      }
    }
  }

  return Changed;
}

bool removeInstanceBackedgesFromInstanceAtOffset0Loops(LayoutTypeSystem &TS) {
  return removeBackedgesFromSCC<InstanceOffsetZeroWithInstanceBackedge>(TS);
}

} // end namespace dla
