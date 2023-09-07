//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
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
#include "llvm/Support/Progress.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"

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
  // lookin only at **undirected** SCCNodeView edges. Each of this subgraphs is
  // called "component". The idea is that SCCNodeView edges are more meaningful
  // than SCCBackedgeView edges, so we don't want to remove any of them, but we
  // need to identify such edges that create loops across multiple components,
  // and cut them.
  llvm::EquivalenceClasses<const LTSN *> Components;
  {
    // Depth first visit across SCCNodeView edges.
    llvm::df_iterator_default_set<const LTSN *, 16> Visited;

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

  using MixedNodeT = EdgeFilteredGraph<LTSN *, isMixedEdge<SCC>>;

  T.advance("Remove Backedges");
  for (const auto &Root : llvm::nodes(&TS)) {
    revng_assert(Root != nullptr);
    // We start from SCCNodeView roots and look if we find an SCC with mixed
    // edges.
    if (hasNoSCCEdge<SCC>(Root))
      continue;

    if (not isSCCRoot<SCC>(Root))
      continue;

    revng_log(Log, "# Looking for mixed loops from: " << Root->ID);

    struct EdgeInfo {
      LTSN *Src;
      LTSN *Tgt;
      const TypeLinkTag *Tag;
      // Comparison operators to use in set
      std::strong_ordering operator<=>(const EdgeInfo &) const = default;
    };

    llvm::SmallPtrSet<const LTSN *, 16> Visited;
    llvm::SmallPtrSet<const LTSN *, 16> InStack;

    struct StackEntry {
      LTSN *Node;
      const LTSN *ComponentLeader;
      typename MixedNodeT::ChildEdgeIteratorType NextToVisitIt;
    };
    std::vector<StackEntry> VisitStack;

    const auto TryPush = [&](LTSN *N, const LTSN *ComponentLeader) {
      revng_log(Log, "--* try_push(" << N->ID << ')');
      LoggerIndent Indent{ Log };
      bool NewVisit = Visited.insert(N).second;
      if (NewVisit) {
        revng_log(Log, "component leader: " << ComponentLeader);

        VisitStack.push_back({ N,
                               ComponentLeader,
                               MixedNodeT::child_edge_begin(N) });
        InStack.insert(N);
        revng_assert(InStack.size() == VisitStack.size());
        revng_log(Log, "--> pushed!");
      } else {
        revng_log(Log, "--| already visited!");
      }
      return NewVisit;
    };

    const auto Pop = [&VisitStack, &InStack]() {
      revng_log(Log, "<-- pop(" << VisitStack.back().Node->ID << ')');
      InStack.erase(VisitStack.back().Node);
      VisitStack.pop_back();
      revng_assert(InStack.size() == VisitStack.size());
    };

    llvm::SmallSet<EdgeInfo, 8> ToRemove;
    llvm::SmallVector<EdgeInfo, 8> CrossComponentEdges;

    revng_assert(Components.findValue(Root) != Components.end());
    TryPush(Root, Components.getLeaderValue(Root));
    while (not VisitStack.empty()) {
      StackEntry &Top = VisitStack.back();

      const LTSN *TopComponent = Top.ComponentLeader;
      LTSN *TopNode = Top.Node;
      typename MixedNodeT::ChildEdgeIteratorType
        &NextEdgeToVisit = Top.NextToVisitIt;

      revng_log(Log,
                "## Stack top has ID: " << TopNode->ID
                                        << " ComponentLeader ID: "
                                        << TopComponent->ID);

      bool StartNew = false;
      while (not StartNew
             and NextEdgeToVisit != MixedNodeT::child_edge_end(TopNode)) {

        LTSN *NextChild = NextEdgeToVisit->first;
        const TypeLinkTag *NextTag = NextEdgeToVisit->second;
        EdgeInfo E = { TopNode, NextChild, NextTag };

        revng_log(Log, "### Next child ID: " << NextChild->ID);

        // Check if the next children is in a component.
        // If it's not, leave the same component of the top of the stack, so
        // that we can identify the first edge that closes the crossing from one
        // component to another.
        const LTSN *NextComponent = TopComponent;
        if (auto ComponentIt = Components.findValue(NextChild);
            ComponentIt != Components.end()) {
          revng_log(Log, "Next is in a Component");
          NextComponent = *Components.findLeader(ComponentIt);
          if (NextComponent != TopComponent) {
            revng_log(Log,
                      "Push Cross-Component Edge " << TopNode->ID << " -> "
                                                   << NextChild->ID);
            revng_assert(SCC::BackedgeNodeView::filter()({ nullptr, E.Tag }));
            CrossComponentEdges.push_back(std::move(E));
          }
        }

        ++NextEdgeToVisit;
        StartNew = TryPush(NextChild, NextComponent);

        if (not StartNew) {

          // We haven't pushed, either because NextChild is on the stack, or
          // because it was visited before.
          if (InStack.contains(NextChild)) {

            // If it's on the stack, we're closing a loop.
            // Add all the cross-component edges to the edges ToRemove.
            revng_log(Log, "Closes Loop");
            if (Log.isEnabled()) {
              for (EdgeInfo &E : CrossComponentEdges) {
                revng_log(Log,
                          "Is to remove: " << E.Src->ID << " -> " << E.Tgt->ID);
              }
            }
            ToRemove.insert(CrossComponentEdges.begin(),
                            CrossComponentEdges.end());

            // This an optimization.
            // All the CrossComponentEdges have just been added to the edges
            // ToRemove, so there's no point keeping them also in
            // CrossComponentEdges, and possibly trying to insert them again
            // later. We can drop all of them here.
            CrossComponentEdges.clear();

            if (NextComponent == TopComponent
                and SCC::BackedgeNodeView::filter()({ nullptr, E.Tag })) {
              // This means that the edge E we tried to push on the stack is an
              // BackedgeNodeView edge closing a loop.
              ToRemove.insert(std::move(E));
            }
          }

          if (NextComponent != TopComponent
              and not CrossComponentEdges.empty()) {
            EdgeInfo E = CrossComponentEdges.pop_back_val();
            revng_log(Log,
                      "Pop Cross-Component Edge " << E.Src->ID << " -> "
                                                  << E.Tgt->ID);
          }
        }
      }

      if (StartNew) {
        // We exited the push loop with a TryPush succeeding, so we need to look
        // at the new child freshly pushed on the stack.
        continue;
      }

      revng_log(Log, "## Completed : " << TopNode->ID);

      Pop();

      if (not VisitStack.empty() and not CrossComponentEdges.empty()
          and TopComponent != VisitStack.back().ComponentLeader) {
        // We are popping back a cross-component edge. Remove it.
        EdgeInfo E = CrossComponentEdges.pop_back_val();
        revng_log(Log,
                  "Pop Cross-Component Edge " << E.Src->ID << " -> "
                                              << E.Tgt->ID);
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
        if (I.hasCycle())
          revng_check(false);
      }
    }
  }

  return Changed;
}

bool removeInstanceBackedgesFromInstanceAtOffset0Loops(LayoutTypeSystem &TS) {
  return removeBackedgesFromSCC<InstanceOffsetZeroWithInstanceBackedge>(TS);
}

} // end namespace dla
