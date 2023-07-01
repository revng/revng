//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <compare>
#include <limits>
#include <vector>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"

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

  // Color all the nodes, except those that have no incoming nor outgoing
  // SCCNodeView edges.
  // The goal is to identify the subsets of nodes that are connected by means of
  // SCCNodeView edges.
  // In this way we divide the graph in subgraphs, such that for each pair of
  // nodes P and Q with (P != Q) in the same sugraphs (i.e. with the same
  // color), either P is reachable from Q, or Q is reachable from P (even if not
  // at step one), by means of SCCNodeView edges.
  // Each of this subgraphs is called "component".
  // The idea is that SCCNodeView edges are more meaningful than SCCBackedgeView
  // edges, so we don't want to remove any of them, but we need to identify
  // suc edges that create loops across multiple components, and cut them.
  std::map<const LTSN *, unsigned> NodeColors;
  {
    // Holds a set of nodes.
    using NodeSet = llvm::df_iterator_default_set<const LTSN *, 16>;

    // Map colors to set of nodes with that color.
    std::map<unsigned, NodeSet> ColorToNodes;
    unsigned NewColor = 0UL;

    revng_log(Log, "Detect colors");
    for (const LTSN *Root : llvm::nodes(&TS)) {
      revng_assert(Root != nullptr);
      revng_log(Log, "Root->ID: " << Root->ID);
      // Skip nodes that have no incoming or outgoing SCCNodeView edges.
      if (hasNoSCCEdge<SCC>(Root))
        continue;
      // Start visiting only from roots of SCCNodeView edges.
      if (not isSCCRoot<SCC>(Root))
        continue;

      revng_log(Log, "DFS from Root->ID: " << Root->ID);
      revng_log(Log, "NewColor: " << NewColor);
      LoggerIndent Indent{ Log };

      // Depth first visit across SCCNodeView edges.
      llvm::df_iterator_default_set<const LTSN *, 16> Visited;
      // Tracks the set of colors we found during this visit.
      llvm::SmallSet<unsigned, 16> FoundColors;
      for (auto *N :
           llvm::depth_first_ext(typename SCC::SCCNodeView(Root), Visited)) {
        revng_log(Log, "N->ID: " << N->ID);
        LoggerIndent MoreIndent{ Log };
        // If N is colored, we have already visited it starting from another
        // Root. We add it to the FoundColors and mark its SCCNodeView children
        // as visited, so that they are skipped in the depth first visit.
        if (auto NodeColorIt = NodeColors.find(N);
            NodeColorIt != NodeColors.end()) {
          unsigned Color = NodeColorIt->second;
          revng_log(Log, "already colored - color: " << Color);
          FoundColors.insert(Color);
          for (const LTSN *Child :
               llvm::children<typename SCC::SCCNodeView>(N)) {
            LoggerIndent MoreMoreIndent{ Log };
            revng_log(Log, "push Child->ID: " << Child->ID);
            Visited.insert(Child);
          }
        } else {
          revng_log(Log, "not colored");
        }
      }

      // Add the visited nodes to the ColorToNodesMap, with a new color.
      auto It = ColorToNodes.insert({ NewColor, std::move(Visited) }).first;
      // If we encountered other colors during the visit, all the merged colors
      // need to be merged into the new color.
      if (not FoundColors.empty()) {
        llvm::SmallVector<decltype(ColorToNodes)::iterator, 8> OldToErase;
        // Merge all the sets of nodes with the colors we found with the new
        // set of nodes with the new color.
        for (unsigned OldColor : FoundColors) {
          auto ColorToNodesIt = ColorToNodes.find(OldColor);
          revng_assert(ColorToNodesIt != ColorToNodes.end());
          auto &OldColoredNodes = ColorToNodesIt->second;
          It->second.insert(OldColoredNodes.begin(), OldColoredNodes.end());
          // Mark this iterator as OldToErase, because after we're done merging
          // the old color sets need to be dropped.
          OldToErase.push_back(ColorToNodesIt);
        }

        // Drop the set of nodes with old colors.
        for (auto &ColorToNodesIt : OldToErase)
          ColorToNodes.erase(ColorToNodesIt);
      }

      // Set the proper color to all the newly found nodes.
      for (auto *Node : It->second)
        NodeColors[Node] = NewColor;

      ++NewColor;
    }
  }

  // Here all the nodes are colored.
  // Each component has a different color, while nodes that have no incoming or
  // outgoing SCCNodeView edges do not have a color.

  using MixedNodeT = EdgeFilteredGraph<LTSN *, isMixedEdge<SCC>>;

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
      unsigned Color;
      typename MixedNodeT::ChildEdgeIteratorType NextToVisitIt;
    };
    std::vector<StackEntry> VisitStack;

    const auto TryPush = [&](LTSN *N, unsigned Color) {
      revng_log(Log, "--* try_push(" << N->ID << ')');
      bool NewVisit = Visited.insert(N).second;
      if (NewVisit) {
        revng_log(Log, "    color: " << Color);
        revng_assert(Color != std::numeric_limits<unsigned>::max());

        VisitStack.push_back({ N, Color, MixedNodeT::child_edge_begin(N) });
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
    llvm::SmallVector<EdgeInfo, 8> CrossColorEdges;

    TryPush(Root, NodeColors.at(Root));
    while (not VisitStack.empty()) {
      StackEntry &Top = VisitStack.back();

      unsigned TopColor = Top.Color;
      LTSN *TopNode = Top.Node;
      typename MixedNodeT::ChildEdgeIteratorType
        &NextEdgeToVisit = Top.NextToVisitIt;

      revng_log(Log,
                "## Stack top is: " << TopNode->ID
                                    << "\n          color: " << TopColor);

      bool StartNew = false;
      while (not StartNew
             and NextEdgeToVisit != MixedNodeT::child_edge_end(TopNode)) {

        LTSN *NextChild = NextEdgeToVisit->first;
        const TypeLinkTag *NextTag = NextEdgeToVisit->second;
        EdgeInfo E = { TopNode, NextChild, NextTag };

        revng_log(Log, "### Next child:: " << NextChild->ID);

        // Check if the next children is colored.
        // If it's not, leave the same color of the top of the stack, so that we
        // can identify the first edge that closes the crossing from one
        // component to another.
        unsigned NextColor = TopColor;
        if (auto ColorsIt = NodeColors.find(NextChild);
            ColorsIt != NodeColors.end()) {
          revng_log(Log, "Colored");
          NextColor = ColorsIt->second;
          if (NextColor != TopColor) {
            revng_log(Log,
                      "Push Cross-Color Edge " << TopNode->ID << " -> "
                                               << NextChild->ID);
            revng_assert(SCC::BackedgeNodeView::filter()({ nullptr, E.Tag }));
            CrossColorEdges.push_back(std::move(E));
          }
        }

        ++NextEdgeToVisit;
        StartNew = TryPush(NextChild, NextColor);

        if (not StartNew) {

          // We haven't pushed, either because NextChild is on the stack, or
          // because it was visited before.
          if (InStack.contains(NextChild)) {

            // If it's on the stack, we're closing a loop.
            // Add all the cross color edges to the edges ToRemove.
            revng_log(Log, "Closes Loop");
            if (Log.isEnabled()) {
              for (EdgeInfo &E : CrossColorEdges) {
                revng_log(Log,
                          "Is to remove: " << E.Src->ID << " -> " << E.Tgt->ID);
              }
            }
            ToRemove.insert(CrossColorEdges.begin(), CrossColorEdges.end());

            // This an optimization.
            // All the CrossColorEdges have just been added to the edges
            // ToRemove, so there's no point keeping them also in
            // CrossColorEdges, and possibly trying to insert them again later.
            // We can drop all of them here.
            CrossColorEdges.clear();

            if (NextColor == TopColor
                and SCC::BackedgeNodeView::filter()({ nullptr, E.Tag })) {
              // This means that the edge E we tried to push on the stack is an
              // BackedgeNodeView edge closing a loop.
              ToRemove.insert(std::move(E));
            }
          }

          if (NextColor != TopColor and not CrossColorEdges.empty()) {
            EdgeInfo E = CrossColorEdges.pop_back_val();
            revng_log(Log,
                      "Pop Cross-Color Edge " << E.Src->ID << " -> "
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

      if (not VisitStack.empty() and not CrossColorEdges.empty()
          and TopColor != VisitStack.back().Color) {
        // We are popping back a cross-color edge. Remove it.
        EdgeInfo E = CrossColorEdges.pop_back_val();
        revng_log(Log,
                  "Pop Cross-Color Edge " << E.Src->ID << " -> " << E.Tgt->ID);
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
