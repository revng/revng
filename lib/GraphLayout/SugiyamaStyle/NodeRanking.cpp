/// \file NodeRanking.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"

#include "revng/ADT/SmallMap.h"

#include "NodeRanking.h"

// This is a workaround for the case where back-edge normalization causes new
// nodes to be added before the root node. Since ranks of those nodes has to be
// smaller than the rank of the root node - they loop around causing huge
// numbers to appear.
// As a temporary workaround, the root node rank can be set to a bigger value
// allowing a couple of "free" layers above it to mitigate the problem.
// This is not a perfect solution BUT it should be good enough for any
// reasonable graph.
constexpr Rank RootNodeRankValue = 4;

/// `RankingStrategy::BreadthFirstSearch` template specialization.
/// Assigns ranks based on the BFS visit order.
/// This is the general ranking induced by the graph.
constexpr auto BFS = RankingStrategy::BreadthFirstSearch;
template<>
RankContainer rankNodes<BFS>(InternalGraph &Graph) {
  RankContainer Ranks;

  for (auto *Node : llvm::breadth_first(Graph.getEntryNode())) {
    auto CurrentRank = Ranks.try_emplace(Node, RootNodeRankValue).first->second;
    for (auto *Successor : Node->successors())
      if (auto SuccessorIt = Ranks.find(Successor); SuccessorIt == Ranks.end())
        Ranks.emplace(Successor, CurrentRank + 1);
      else if (SuccessorIt->second < CurrentRank + 1)
        SuccessorIt->second = CurrentRank + 1;
  }

  revng_assert(Ranks.size() == Graph.size());
  return Ranks;
}

/// `RankingStrategy::DepthFirstSearch` template specialization.
/// Assigns ranks based on the DFS time.
constexpr auto DFS = RankingStrategy::DepthFirstSearch;
template<>
RankContainer rankNodes<DFS>(InternalGraph &Graph) {
  RankContainer Ranks;

  auto Counter = RootNodeRankValue;
  for (auto *Node : llvm::depth_first(Graph.getEntryNode()))
    Ranks.try_emplace(Node, Counter++);

  revng_assert(Ranks.size() == Graph.size());
  return Ranks;
}

/// `RankingStrategy::Topological` template specialization.
/// Assigns ranks based on the DFS topological order.
///
/// This ranking can be used to get a layout where each node is on
/// its own layer. The idea is that it produces a layout with a single
/// real node per layer while going along a path before visiting the branches,
/// in opposition to the conventional BFS-based topological order, where
/// all the branches are visited at the same time
///
/// The goal of this ranking is to produce something similar to what ghidra
/// does without using a decompiler.
constexpr auto TRS = RankingStrategy::Topological;
template<>
RankContainer rankNodes<TRS>(InternalGraph &Graph) {
  RankContainer Ranks;

  auto Counter = RootNodeRankValue;
  for (auto *Node : llvm::ReversePostOrderTraversal(Graph.getEntryNode()))
    Ranks.try_emplace(Node, Counter++);

  revng_assert(Ranks.size() == Graph.size());
  return Ranks;
}

/// Checks whether two nodes are disjoint provided pointers to the nodes
/// and the lists of successors for each and every node.
///
/// It also detects small diamonds with depth around 3.
///
/// The expected number of branches is around 2 so the total number of visited
/// nodes should be less than 16 on average
static bool areDisjoint(NodeView A, NodeView B, size_t MaxDepth = -1) {
  std::deque<NodeView> Queue;

  std::vector<NodeView> Worklist;
  Worklist.push_back(A);
  Worklist.push_back(B);

  // Perform a joint BFS and check that there are no shared nodes
  SmallMap<NodeView, size_t, 16> VisitedPrevious;
  for (const auto &Current : Worklist) {
    SmallMap<NodeView, size_t, 16> VisitedNow;
    VisitedNow[Current] = 0;
    if (VisitedPrevious.contains(Current))
      return false;
    Queue.emplace_back(Current);

    while (!Queue.empty()) {
      auto TopNode = Queue.front();
      Queue.pop_front();

      auto VisitedTop = VisitedNow[Current];
      if (VisitedTop > MaxDepth)
        continue;

      for (auto *Next : TopNode->successors()) {
        if (!VisitedNow.contains(Next)) {
          Queue.emplace_back(Next);
          VisitedNow[Next] = VisitedTop + 1;
        }
      }
    }
    std::swap(VisitedPrevious, VisitedNow);
  }

  return true;
}

/// `RankingStrategy::DisjointDepthFirstSearch` template specialization.
/// Assigns ranks based on the DFS time but allows multiple nodes at the same
/// rank when two paths diverge if they either rejoin shortly or never
/// rejoin again.
constexpr auto DDFS = RankingStrategy::DisjointDepthFirstSearch;
template<>
RankContainer rankNodes<DDFS>(InternalGraph &Graph, int64_t DiamondBound) {
  RankContainer Ranks;

  std::vector<NodeView> Stack;
  for (auto *Node : Graph.nodes()) {
    if (!Node->hasPredecessors() && !Ranks.contains(Node)) {
      Stack.emplace_back(Node);

      size_t DFSTime = RootNodeRankValue;
      while (!Stack.empty()) {
        auto Current = Stack.back();
        Stack.pop_back();

        auto &CurrentRank = Ranks[Current];
        if (CurrentRank == 0)
          CurrentRank = DFSTime++;

        bool IsDisjoint = true;
        bool IsDiamond = true;
        if (Current->successorCount() > 1) {
          auto *First = *Current->successors().begin();
          for (auto NextIt = std::next(Current->successors().begin());
               NextIt != Current->successors().end();
               ++NextIt) {
            auto &DB = DiamondBound;
            IsDisjoint = IsDisjoint && areDisjoint(First, *NextIt);
            IsDiamond = IsDiamond && !areDisjoint(First, *NextIt, DB);
          }
        } else {
          IsDisjoint = IsDiamond = false;
        }

        for (auto *Next : Current->successors()) {
          if (!Ranks.contains(Next)) {
            Stack.push_back(Next);
            if (IsDisjoint || IsDiamond)
              Ranks[Next] = CurrentRank + 1;
            else
              Ranks[Next] = 0;
          }
        }
      }
    }
  }

  revng_assert(Ranks.size() == Graph.size());
  return Ranks;
}

template<>
RankContainer rankNodes<DDFS>(InternalGraph &Graph) {
  return rankNodes<DDFS>(Graph, 4u);
}

RankContainer &updateRanks(InternalGraph &Graph, RankContainer &Ranks) {
  Ranks.try_emplace(Graph.getEntryNode(),
                    Graph.getEntryNode()->IsVirtual ? Rank(-1) : Rank(0));

  for (auto *Current : llvm::ReversePostOrderTraversal(Graph.getEntryNode())) {
    auto &CurrentRank = Ranks[Current];
    for (auto *Predecessor : Current->predecessors())
      if (auto Rank = Ranks[Predecessor]; Rank + 1 > CurrentRank)
        CurrentRank = Rank + 1;
  }

  revng_assert(Ranks.size() == Graph.size());
  return Ranks;
}
