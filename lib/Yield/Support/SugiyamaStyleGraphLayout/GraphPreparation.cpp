/// \file GraphPreparation.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>
#include <vector>

#include "Layout.h"
#include "NodeRanking.h"

// Converts an external graph into an internal one.
static InternalGraph convertToInternal(ExternalGraph &Graph) {
  std::unordered_map<ExternalNode *, InternalNode *> LookupTable;

  // Add all the "external" nodes to the "internal" graph.
  InternalGraph Result;
  for (auto *Node : Graph.nodes())
    LookupTable.emplace(Node, Result.addNode(Node));

  // Add all the "external" edges to the "internal" graph.
  for (auto *From : Graph.nodes())
    for (auto [To, Label] : From->successor_edges())
      LookupTable.at(From)->addSuccessor(LookupTable.at(To),
                                         InternalLabel{ Label });

  return Result;
}

// A simple container that's used to indicate a self-loop.
struct SelfLoop {
  InternalNode *Node;
  ExternalLabel *Label;

  SelfLoop(InternalNode *Node, ExternalLabel *Label) :
    Node(Node), Label(Label) {}
};
using SelfLoopContainer = llvm::SmallVector<SelfLoop, 16>;

// Removes self-loops from the graph and returns their labels.
static SelfLoopContainer extractSelfLoops(InternalGraph &Graph) {
  SelfLoopContainer Result;

  for (auto *Node : Graph.nodes()) {
    for (auto Iterator = Node->successor_edges().begin();
         Iterator != Node->successor_edges().end();) {
      if (Iterator->Neighbor->Index == Node->Index) {
        Result.emplace_back(Node, Iterator->Label->Pointer);
        Iterator = Node->removeSuccessor(Iterator);
      } else {
        ++Iterator;
      }
    }
  }

  return Result;
}

/// Returns a list of backwards edges of the graph and the order
/// they were visited in.
/// \note: this function could use a rework.
static std::pair<std::set<EdgeView>, std::vector<NodeView>>
pickBackwardsEdges(InternalGraph &Graph) {
  std::set<EdgeView> Result;

  std::unordered_set<NodeView> Visited, OnStack;
  std::vector<NodeView> VisitOrder;

  for (auto *Node : Graph.nodes()) {
    if (!Visited.contains(Node)) {
      std::vector<std::pair<NodeView, Rank>> Stack{ { Node, 0 } };
      while (!Stack.empty()) {
        auto [Current, Rank] = Stack.back();
        Stack.pop_back();

        if (Rank == 0) {
          VisitOrder.push_back(Current);
          Visited.emplace(Current);
          OnStack.emplace(Current);
        }

        bool Finished = true;
        for (auto It = std::next(Current->successor_edges().begin(), Rank);
             It != Current->successor_edges().end();
             ++It) {
          auto [Neighbor, Label] = *It;
          if (!Visited.contains(Neighbor)) {
            Stack.emplace_back(Current, Rank + 1);
            Stack.emplace_back(Neighbor, 0);
            Visited.emplace(Neighbor);
            Finished = false;
            break;
          } else if (OnStack.contains(Neighbor))
            Result.emplace(Current, Neighbor, Label);
        }

        if (Finished) {
          revng_assert(OnStack.contains(Current));
          OnStack.erase(Current);
        }
      }
    }
  }

  revng_assert(VisitOrder.size() == Graph.size());
  return { Result, VisitOrder };
}

/// Reverses the edge (its `From` node becomes its `To` node and vice versa).
static void invertEdge(EdgeView Edge) {
  revng_assert(Edge.Label != nullptr);

  Edge.Label->IsBackwards = !Edge.Label->IsBackwards;
  Edge.To->addSuccessor(Edge.From, std::move(*Edge.Label));

  auto Comparator = [&Edge](auto SuccessorEdge) {
    return SuccessorEdge.Neighbor == Edge.To
           && SuccessorEdge.Label == Edge.Label;
  };
  auto Iterator = llvm::find_if(Edge.From->successor_edges(), Comparator);
  revng_assert(Iterator != Edge.From->successor_edges().end(),
               "Unable to reverse an edge that doesn't exist");
  Edge.From->removeSuccessor(Iterator);
}

/// Ensures an "internal" graph to be a DAG by "flipping" edges to prevent
/// loops.
static void convertToDAG(InternalGraph &Graph) {
  if (auto Initial = pickBackwardsEdges(Graph).first; !Initial.empty()) {
    // Iteratively reverse all the initial backedges.
    for (auto &&Edge : Initial)
      invertEdge(std::move(Edge));

    // There is a rare corner case where flipping an edge causes a change in
    // the DFS order that results in new backedges. In that case we enforce
    // acyclic orientation. But, it may be suboptimal: some of the forward
    // edges are reversed as well.
    if (auto [Edges, Order] = pickBackwardsEdges(Graph); !Edges.empty()) {
      std::unordered_map<NodeView, Index> Lookup;
      for (Index I = 0; I < Order.size(); ++I)
        Lookup.emplace(Order.at(I), I);

      std::vector<EdgeView> ToReverse;
      for (auto *From : Graph.nodes())
        for (auto [To, Label] : From->successor_edges())
          if (From->hasPredecessors() && Lookup.at(From) > Lookup.at(To))
            ToReverse.emplace_back(From, To, Label);

      for (auto &&Edge : ToReverse)
        invertEdge(std::move(Edge));

      if (!pickBackwardsEdges(Graph).first.empty()) {
        for (auto *From : Graph.nodes())
          for (auto Iterator = From->successors().begin();
               Iterator != From->successors().end();)
            if (From->Index == (*Iterator)->Index)
              Iterator = From->removeSuccessor(Iterator);
            else
              ++Iterator;
      }

      revng_assert(pickBackwardsEdges(Graph).first.empty());
    }
  }
}

// Calculates the absolute difference in rank between two nodes.
// In other words, the result represents the number of layers the edge between
// the specified two nodes needs to go through.
static RankDelta delta(NodeView LHS, NodeView RHS, const RankContainer &Ranks) {
  return std::abs(RankDelta(Ranks.at(LHS)) - RankDelta(Ranks.at(RHS)));
}

// Returns a list of edges that span accross more than a single layer.
static std::vector<EdgeView>
pickLongEdges(InternalGraph &Graph, const RankContainer &Ranks) {
  std::vector<EdgeView> Result;

  for (auto *From : Graph.nodes())
    for (auto [To, Label] : From->successor_edges())
      if (delta(From, To, Ranks) > RankDelta(1))
        Result.emplace_back(From, To, Label);

  return Result;
}

template<typename EdgeType, RankingStrategy Strategy>
void partition(const std::vector<EdgeType> &Edges,
               InternalGraph &Graph,
               const RankContainer &Ranks,
               NodeClassifier<Strategy> &Classifier) {
  for (auto &Edge : Edges) {
    size_t PartitionCount = delta(Edge.From, Edge.To, Ranks);

    auto Current = Edge.From;
    if (PartitionCount != 0) {
      for (size_t Partition = 0; Partition < PartitionCount - 1; ++Partition) {
        auto NewNode = Graph.addNode(nullptr);

        const InternalLabel &LabelCopy = Edge.label();
        Current->addSuccessor(NewNode, LabelCopy);
        Classifier.addLongEdgePartition(Current, NewNode);

        Current = NewNode;
      }
    }

    const InternalLabel &LabelCopy = Edge.label();
    Current->addSuccessor(Edge.To, LabelCopy);
    Classifier.addLongEdgePartition(Current, Edge.To);
  }
}

/// Breaks long edges into partitions by introducing new internal nodes.
template<RankingStrategy Strategy>
RankContainer
partitionLongEdges(InternalGraph &Graph, NodeClassifier<Strategy> &Classifier) {
  // To simplify the ranking algorithms, if there's more than one entry point,
  // an artifitial entry node is added. This new node has a single edge per
  // real entry point.
  llvm::SmallVector<NodeView, 4> EntryNodes;
  for (auto *Node : Graph.nodes())
    if (!Node->hasPredecessors())
      EntryNodes.emplace_back(Node);
  revng_assert(!EntryNodes.empty());

  revng_assert(Graph.getEntryNode() == nullptr);
  if (EntryNodes.size() > 1) {
    auto EntryPoint = Graph.addNode(nullptr);
    for (auto &Node : EntryNodes)
      EntryPoint->addSuccessor(Node, nullptr);
    Graph.setEntryNode(EntryPoint);
  } else {
    Graph.setEntryNode(EntryNodes.front());
  }

  // Because a long edge can also be a backwards edge, edges that are certainly
  // long need to be removed first, so that DFS-based rankind algorithms don't
  // mistakenly take any undesired shortcuts. Simple BFS ranking used
  // internally to differencite such "certainly long" edges.

  // Rank nodes based on a BreadthFirstSearch pass-through.
  auto Ranks = rankNodes<RankingStrategy::BreadthFirstSearch>(Graph);

  /// A copy of an edge label.
  using EdgeCopy = detail::GenericEdgeView<InternalLabel>;

  // Temporary save them outside of the graph.
  std::vector<EdgeCopy> SavedLongEdges;
  for (auto *From : Graph.nodes()) {
    for (auto Iterator = From->successor_edges_rbegin();
         Iterator != From->successor_edges_rend();) {
      if (auto [To, Label] = *Iterator; delta(From, To, Ranks) > RankDelta(1)) {
        revng_assert(Label != nullptr);
        SavedLongEdges.emplace_back(From, To, std::move(*Label));
        Iterator = From->removeSuccessor(Iterator);
      } else {
        ++Iterator;
      }
    }
  }

  // Calculate real ranks for the remainder of the graph.
  Ranks = rankNodes<Strategy>(Graph);

  // Pick new long edges based on the real ranks.
  auto NewLongEdges = pickLongEdges(Graph, Ranks);

  // Add partitions based on removed earlier edges.
  partition(SavedLongEdges, Graph, Ranks, Classifier);

  // Add partitions based on the new long edges.
  partition(NewLongEdges, Graph, Ranks, Classifier);
  for (auto &Edge : NewLongEdges)
    Edge.From->removeSuccessors(Edge.To);

  // Now, make the DFS ranking consistent by checking that the rank of a node
  // is greater than the rank of its predecessors.
  //
  // Eventually, this ranking score becomes a proper hierarchy.
  updateRanks(Graph, Ranks);

  // Make sure that new long edges are properly broken up.
  NewLongEdges = pickLongEdges(Graph, Ranks);
  partition(NewLongEdges, Graph, Ranks, Classifier);
  for (auto &Edge : NewLongEdges)
    Edge.From->removeSuccessors(Edge.To);

  updateRanks(Graph, Ranks);

  // Remove an artificial entry node if it was added.
  if (Graph.hasEntryNode && Graph.getEntryNode() != nullptr) {
    if (Graph.getEntryNode()->isVirtual()) {
      Ranks.erase(Graph.getEntryNode());
      Graph.removeNode(Graph.getEntryNode());
    }
  }

  revng_assert(pickLongEdges(Graph, Ranks).empty());
  return Ranks;
}

template<RankingStrategy Strategy>
void partitionArtificialBackwardsEdges(InternalGraph &Graph,
                                       RankContainer &Ranks,
                                       NodeClassifier<Strategy> &Classifier) {
  for (size_t NodeIndex = 0; NodeIndex < Graph.size(); ++NodeIndex) {
    auto *From = *std::next(Graph.nodes().begin(), NodeIndex);
    for (auto EdgeIterator = From->successor_edges_rbegin();
         EdgeIterator != From->successor_edges_rend();) {
      auto [To, Label] = *EdgeIterator;
      if (From->isVirtual() != To->isVirtual() && Label->IsBackwards == true) {
        auto *LabelPointer = Label->Pointer;
        EdgeIterator = From->removeSuccessor(EdgeIterator);

        auto *NewNode1 = Graph.addNode(nullptr);
        auto *NewNode2 = Graph.addNode(nullptr);

        if (From->isVirtual() && !To->isVirtual()) {
          // Fix the low point of a backwards edge
          From->addSuccessor(NewNode2, InternalLabel(LabelPointer, true));
          NewNode2->addSuccessor(NewNode1, InternalLabel(LabelPointer, true));
          To->addSuccessor(NewNode1, InternalLabel(LabelPointer, false));

          Classifier.addBackwardsEdgePartition(From, NewNode2);
          Classifier.addBackwardsEdgePartition(NewNode2, NewNode1);
          Classifier.addBackwardsEdgePartition(To, NewNode1);

          Ranks[NewNode1] = Ranks.at(To) + 1;
          Ranks[NewNode2] = Ranks.at(To);
        } else {
          // Fix the high point of a backwards edge
          NewNode1->addSuccessor(From, InternalLabel(LabelPointer, false));
          NewNode1->addSuccessor(NewNode2, InternalLabel(LabelPointer, true));
          NewNode2->addSuccessor(To, InternalLabel(LabelPointer, true));

          Classifier.addBackwardsEdgePartition(NewNode1, From);
          Classifier.addBackwardsEdgePartition(NewNode1, NewNode2);
          Classifier.addBackwardsEdgePartition(NewNode2, To);

          Ranks[NewNode1] = Ranks.at(From) - 1;
          Ranks[NewNode2] = Ranks.at(From);
        }
      } else {
        ++EdgeIterator;
      }
    }
  }
}

template<RankingStrategy Strategy>
void partitionOriginalBackwardsEdges(InternalGraph &Graph,
                                     RankContainer &Ranks,
                                     NodeClassifier<Strategy> &Classifier) {
  for (size_t NodeIndex = 0; NodeIndex < Graph.size(); ++NodeIndex) {
    auto *From = *std::next(Graph.nodes().begin(), NodeIndex);
    for (auto EdgeIterator = From->successor_edges_rbegin();
         EdgeIterator != From->successor_edges_rend();) {
      auto [To, Label] = *EdgeIterator;
      if (!From->isVirtual() && !To->isVirtual()
          && Label->IsBackwards == true) {
        auto *LabelPointer = Label->Pointer;
        EdgeIterator = From->removeSuccessor(EdgeIterator);

        auto *NewNode1 = Graph.addNode(nullptr);
        auto *NewNode2 = Graph.addNode(nullptr);
        auto *NewNode3 = Graph.addNode(nullptr);
        auto *NewNode4 = Graph.addNode(nullptr);

        NewNode1->addSuccessor(From, InternalLabel(LabelPointer, false));
        NewNode1->addSuccessor(NewNode2, InternalLabel(LabelPointer, true));
        NewNode2->addSuccessor(NewNode3, InternalLabel(LabelPointer, true));
        NewNode3->addSuccessor(NewNode4, InternalLabel(LabelPointer, true));
        To->addSuccessor(NewNode4, InternalLabel(LabelPointer, false));

        Classifier.addBackwardsEdgePartition(NewNode1, From);
        Classifier.addBackwardsEdgePartition(NewNode1, NewNode2);
        Classifier.addBackwardsEdgePartition(NewNode2, NewNode3);
        Classifier.addBackwardsEdgePartition(NewNode3, NewNode4);
        Classifier.addBackwardsEdgePartition(To, NewNode4);

        Ranks[NewNode1] = Ranks.at(From) - 1;
        Ranks[NewNode2] = Ranks.at(From);
        Ranks[NewNode3] = Ranks.at(To);
        Ranks[NewNode4] = Ranks.at(To) + 1;
      } else {
        ++EdgeIterator;
      }
    }
  }
}

template<RankingStrategy Strategy>
void partitionSelfLoops(InternalGraph &Graph,
                        RankContainer &Ranks,
                        SelfLoopContainer &SelfLoops,
                        NodeClassifier<Strategy> &Classifier) {
  for (auto &Edge : SelfLoops) {
    auto *NewNode1 = Graph.addNode(nullptr);
    auto *NewNode2 = Graph.addNode(nullptr);
    auto *NewNode3 = Graph.addNode(nullptr);

    NewNode1->addSuccessor(Edge.Node, InternalLabel(Edge.Label, false));
    NewNode1->addSuccessor(NewNode2, InternalLabel(Edge.Label, true));
    NewNode2->addSuccessor(NewNode3, InternalLabel(Edge.Label, true));
    Edge.Node->addSuccessor(NewNode3, InternalLabel(Edge.Label, false));

    Classifier.addBackwardsEdgePartition(NewNode1, Edge.Node);
    Classifier.addBackwardsEdgePartition(NewNode1, NewNode2);
    Classifier.addBackwardsEdgePartition(NewNode2, NewNode3);
    Classifier.addBackwardsEdgePartition(Edge.Node, NewNode3);

    Ranks[NewNode1] = Ranks.at(Edge.Node) - 1;
    Ranks[NewNode2] = Ranks.at(Edge.Node);
    Ranks[NewNode3] = Ranks.at(Edge.Node) + 1;
  }
}

template<RankingStrategy Strategy>
std::tuple<InternalGraph, RankContainer, NodeClassifier<Strategy>>
prepareGraph(ExternalGraph &Graph) {
  // Get the internal representation of the graph.
  InternalGraph Result = convertToInternal(Graph);

  // Temporarily remove self-loops from the graph.
  auto SelfLoops = extractSelfLoops(Result);

  // Temporarily reverse edges so the graph doesn't contain loops.
  convertToDAG(Result);

  // Robust node classification simplifies the permutation selection.
  NodeClassifier<Strategy> Classifier;

  // Split long edges into 1 rank wide partitions.
  auto Ranks = partitionLongEdges(Result, Classifier);

  // Split backwards facing edges created when partitioning the long edges up.
  partitionArtificialBackwardsEdges(Result, Ranks, Classifier);

  // Split the backwards facing edges from the "external" graph into partitions.
  partitionOriginalBackwardsEdges(Result, Ranks, Classifier);

  // Add the self-loops back in a partitioned form.
  partitionSelfLoops(Result, Ranks, SelfLoops, Classifier);

  return { std::move(Result), std::move(Ranks), std::move(Classifier) };
}

template<RankingStrategy Strategy>
using RV = std::tuple<InternalGraph, RankContainer, NodeClassifier<Strategy>>;

template RV<RankingStrategy::BreadthFirstSearch>
prepareGraph<RankingStrategy::BreadthFirstSearch>(ExternalGraph &Graph);

template RV<RankingStrategy::DepthFirstSearch>
prepareGraph<RankingStrategy::DepthFirstSearch>(ExternalGraph &Graph);

template RV<RankingStrategy::Topological>
prepareGraph<RankingStrategy::Topological>(ExternalGraph &Graph);

template RV<RankingStrategy::DisjointDepthFirstSearch>
prepareGraph<RankingStrategy::DisjointDepthFirstSearch>(ExternalGraph &Graph);
