/// \file GraphPreparation.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>
#include <vector>

#include "revng/Support/GraphAlgorithms.h"

#include "InternalCompute.h"
#include "NodeRanking.h"

// A simple container that's used to indicate a self-loop.
struct SelfLoop {
  InternalNode *Node;
  InternalEdge Edge;

  SelfLoop(InternalNode *Node, InternalEdge &&Edge) :
    Node(Node), Edge(std::move(Edge)) {}
};
using SelfLoopContainer = llvm::SmallVector<SelfLoop, 16>;

// Removes self-loops from the graph and returns their labels.
static SelfLoopContainer extractSelfLoops(InternalGraph &Graph) {
  SelfLoopContainer Result;

  for (auto *Node : Graph.nodes()) {
    for (auto Iterator = Node->successor_edges().begin();
         Iterator != Node->successor_edges().end();) {
      if (Iterator->Neighbor->index() == Node->index()) {
        Result.emplace_back(Node, std::move(*Iterator->Label));
        Iterator = Node->removeSuccessor(Iterator);
      } else {
        ++Iterator;
      }
    }
  }

  return Result;
}

/// To simplify the ranking algorithms, if there's more than one entry point,
/// an artificial entry node is added. This new node has a single edge per
/// real entry point.
static void
ensureSingleEntry(InternalGraph &Graph, RankContainer *MaybeRanks = nullptr) {
  auto EntryNodes = entryPoints(&Graph);
  revng_assert(!EntryNodes.empty());

  if (EntryNodes.size() == 1) {
    // If there's only a single entry point, make sure it's set.
    Graph.setEntryNode(EntryNodes.front());
  } else {
    // If there's more than one, add a new virtual node with all the real
    // entry nodes as its direct successors.

    // BUT if the currently set entry node is already virtual, remove it first:
    // this prevents the possibility of chaining virtual entry nodes when this
    // function is invoked on a slightly-modified graph multiple times.
    if (Graph.getEntryNode() != nullptr) {
      if (Graph.getEntryNode()->IsVirtual) {
        Graph.removeNode(Graph.getEntryNode());
        if (MaybeRanks != nullptr)
          MaybeRanks->erase(Graph.getEntryNode());
      }
    }

    InternalNode *EntryPoint = Graph.makeVirtualNode();
    for (InternalNode *Node : Graph.nodes())
      if (!Node->hasPredecessors() && Node->index() != EntryPoint->index())
        Graph.makeEdge(EntryPoint, Node);
    Graph.setEntryNode(EntryPoint);
  }
}

/// Ensures an "internal" graph to be a DAG by "flipping" edges to prevent
/// loops.
static void convertToDAG(InternalGraph &Graph) {
  ensureSingleEntry(Graph);

  for (auto [From, To] : getBackedges(&Graph)) {
    for (auto Iterator = From->successor_edges_begin();
         Iterator != From->successor_edges_end();) {
      if (To == Iterator->Neighbor) {
        revng_assert(Iterator->Label != nullptr);

        Iterator->Label->IsBackwards = !Iterator->Label->IsBackwards;
        To->addSuccessor(From, std::move(*Iterator->Label));

        Iterator = From->removeSuccessor(Iterator);
      } else {
        ++Iterator;
      }
    }
  }

  revng_assert(getBackedges(&Graph).empty());
}

// Calculates the absolute difference in rank between two nodes.
// In other words, the result represents the number of layers the edge between
// the specified two nodes needs to go through.
static RankDelta delta(NodeView LHS, NodeView RHS, const RankContainer &Ranks) {
  return std::abs(RankDelta(Ranks.at(LHS)) - RankDelta(Ranks.at(RHS)));
}

// Returns a list of edges that span across more than a single layer.
static std::vector<EdgeView>
pickLongEdges(InternalGraph &Graph, const RankContainer &Ranks) {
  std::vector<EdgeView> Result;

  for (auto *From : Graph.nodes())
    for (auto [To, Label] : From->successor_edges())
      if (delta(From, To, Ranks) > RankDelta(1))
        Result.emplace_back(From, To, *Label);

  return Result;
}

template<typename EdgeType, RankingStrategy Strategy>
void partition(std::vector<EdgeType> &Edges,
               InternalGraph &Graph,
               const RankContainer &Ranks,
               MaybeClassifier<Strategy> &Classifier) {
  for (auto &Edge : Edges) {
    size_t PartitionCount = delta(Edge.From, Edge.To, Ranks);
    InternalEdge &Label = Edge.label();

    auto Current = Edge.From;
    if (PartitionCount != 0) {
      for (size_t Partition = 0; Partition < PartitionCount - 1; ++Partition) {
        auto *NewNode = Graph.makeVirtualNode();
        Current->addSuccessor(NewNode, Graph.makeVirtualEdge(Label));

        if (Classifier.has_value())
          Classifier->addLongEdgePartition(Current, NewNode);

        Current = NewNode;
      }
    }

    Current->addSuccessor(Edge.To, std::move(Label));

    if (Classifier.has_value())
      Classifier->addLongEdgePartition(Current, Edge.To);
  }
}

/// Breaks long edges into partitions by introducing new internal nodes.
template<RankingStrategy Strategy>
RankContainer partitionLongEdges(InternalGraph &Graph,
                                 MaybeClassifier<Strategy> &Classifier) {
  // The current partitioning algorithm can only work on graphs that allow
  // specifying the entry point in a mutable way.
  static_assert(InternalGraph::hasEntryNode == true);

  ensureSingleEntry(Graph);

  // Helper lambda for graph verification.
  auto HasSingleEntryPoint = [](const InternalGraph &Graph) -> bool {
    auto HasNoPredecessors = [](const InternalGraph::Node *Node) -> bool {
      return !Node->predecessorCount();
    };
    if (llvm::count_if(Graph.nodes(), HasNoPredecessors) != 1)
      return false;

    if (auto Iterator = llvm::find_if(Graph.nodes(), HasNoPredecessors);
        Iterator == Graph.nodes().end() || *Iterator != Graph.getEntryNode()) {
      return false;
    }

    return true;
  };

  // Because a long edge can also be a backwards edge, edges that are certainly
  // long need to be removed first, so that DFS-based ranking algorithms don't
  // mistakenly take any undesired shortcuts. Simple BFS ranking used
  // internally to differentiate such "certainly long" edges.

  // Rank nodes based on a BreadthFirstSearch pass-through.
  revng_assert(HasSingleEntryPoint(Graph));
  auto Ranks = rankNodes<RankingStrategy::BreadthFirstSearch>(Graph);

  // Temporary save them outside of the graph.
  struct SavedEdge : EdgeView {
    InternalEdge Label;

    SavedEdge(NodeView From, NodeView To, InternalEdge &&Label) :
      EdgeView(From, To, Label), Label(std::move(Label)) {}

    InternalEdge &label() { return Label; }
    const InternalEdge &label() const { return Label; }
  };
  std::vector<SavedEdge> SavedLongEdges;
  for (auto *From : Graph.nodes()) {
    for (auto Iterator = From->successor_edges_rbegin();
         Iterator != From->successor_edges_rend();) {
      if (auto [To, Edge] = *Iterator; delta(From, To, Ranks) > RankDelta(1)) {
        revng_assert(Edge != nullptr);
        SavedLongEdges.emplace_back(From, To, std::move(*Edge));
        Iterator = From->removeSuccessor(Iterator);
      } else {
        ++Iterator;
      }
    }
  }

  // Calculate real ranks for the remainder of the graph.
  ensureSingleEntry(Graph, &Ranks);
  revng_assert(HasSingleEntryPoint(Graph));
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
  revng_assert(HasSingleEntryPoint(Graph));
  updateRanks(Graph, Ranks);

  // Make sure that new long edges are properly broken up.
  NewLongEdges = pickLongEdges(Graph, Ranks);
  partition(NewLongEdges, Graph, Ranks, Classifier);
  for (auto &Edge : NewLongEdges)
    Edge.From->removeSuccessors(Edge.To);

  revng_assert(HasSingleEntryPoint(Graph));
  updateRanks(Graph, Ranks);

  // Remove an artificial entry node if it was ever added.
  revng_assert(HasSingleEntryPoint(Graph));
  if (Graph.getEntryNode() != nullptr) {
    if (Graph.getEntryNode()->IsVirtual) {
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
                                       MaybeClassifier<Strategy> &Classifier) {
  for (size_t NodeIndex = 0; NodeIndex < Graph.size(); ++NodeIndex) {
    auto *From = *std::next(Graph.nodes().begin(), NodeIndex);
    for (auto EdgeIterator = From->successor_edges_rbegin();
         EdgeIterator != From->successor_edges_rend();) {
      auto [To, Original] = *EdgeIterator;
      if (From->IsVirtual != To->IsVirtual && Original->IsBackwards == true) {
        // Move the label out, so that the original edge can be deleted right
        // away. If this is not done, inserting new edges might cause
        // the reallocation of the underlying vector - leading to invalidation
        // of `EdgeIterator`.
        InternalEdge Label = std::move(*Original);
        EdgeIterator = From->removeSuccessor(EdgeIterator);
        auto *NewNode1 = Graph.makeVirtualNode();
        auto *NewNode2 = Graph.makeVirtualNode();

        if (From->IsVirtual && !To->IsVirtual) {
          // Make sure the "low" corner of the backwards facing edge looks good
          // by splitting it into three nodes building a "v"-shape.
          To->addSuccessor(NewNode1, Graph.makeVirtualEdge(Label, false));
          NewNode2->addSuccessor(NewNode1, Graph.makeVirtualEdge(Label, true));
          From->addSuccessor(NewNode2, std::move(Label));

          if (Classifier.has_value()) {
            Classifier->addBackwardsEdgePartition(From, NewNode2);
            Classifier->addBackwardsEdgePartition(NewNode2, NewNode1);
            Classifier->addBackwardsEdgePartition(To, NewNode1);
          }

          Ranks[NewNode1] = Ranks.at(To) + 1;
          Ranks[NewNode2] = Ranks.at(To);
        } else {
          // Make sure the "high" corner of the backwards facing edge looks good
          // by splitting it into three nodes building a "v"-shape.
          NewNode1->addSuccessor(From, Graph.makeVirtualEdge(Label, false));
          NewNode1->addSuccessor(NewNode2, Graph.makeVirtualEdge(Label, true));
          NewNode2->addSuccessor(To, std::move(Label));

          if (Classifier.has_value()) {
            Classifier->addBackwardsEdgePartition(NewNode1, From);
            Classifier->addBackwardsEdgePartition(NewNode1, NewNode2);
            Classifier->addBackwardsEdgePartition(NewNode2, To);
          }

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
                                     MaybeClassifier<Strategy> &Classifier) {
  for (size_t NodeIndex = 0; NodeIndex < Graph.size(); ++NodeIndex) {
    auto *From = *std::next(Graph.nodes().begin(), NodeIndex);
    for (auto EdgeIterator = From->successor_edges_rbegin();
         EdgeIterator != From->successor_edges_rend();) {
      auto [To, Original] = *EdgeIterator;
      if (!From->IsVirtual && !To->IsVirtual && Original->IsBackwards == true) {
        InternalEdge Label = std::move(*Original);
        EdgeIterator = From->removeSuccessor(EdgeIterator);

        auto *NewNode1 = Graph.makeVirtualNode();
        auto *NewNode2 = Graph.makeVirtualNode();
        auto *NewNode3 = Graph.makeVirtualNode();
        auto *NewNode4 = Graph.makeVirtualNode();

        Label.IsBackwards = false;
        NewNode1->addSuccessor(From, Graph.makeVirtualEdge(Label, false));
        NewNode1->addSuccessor(NewNode2, Graph.makeVirtualEdge(Label, true));
        NewNode2->addSuccessor(NewNode3, Graph.makeVirtualEdge(Label, true));
        NewNode3->addSuccessor(NewNode4, Graph.makeVirtualEdge(Label, true));
        To->addSuccessor(NewNode4, std::move(Label));

        if (Classifier.has_value()) {
          Classifier->addBackwardsEdgePartition(NewNode1, From);
          Classifier->addBackwardsEdgePartition(NewNode1, NewNode2);
          Classifier->addBackwardsEdgePartition(NewNode2, NewNode3);
          Classifier->addBackwardsEdgePartition(NewNode3, NewNode4);
          Classifier->addBackwardsEdgePartition(To, NewNode4);
        }

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
                        MaybeClassifier<Strategy> &Classifier) {
  for (auto &&[Node, Edge] : SelfLoops) {
    auto *NewNode1 = Graph.makeVirtualNode();
    auto *NewNode2 = Graph.makeVirtualNode();
    auto *NewNode3 = Graph.makeVirtualNode();

    Edge.IsBackwards = false;
    NewNode1->addSuccessor(Node, Graph.makeVirtualEdge(Edge, false));
    NewNode1->addSuccessor(NewNode2, Graph.makeVirtualEdge(Edge, true));
    NewNode2->addSuccessor(NewNode3, Graph.makeVirtualEdge(Edge, true));
    Node->addSuccessor(NewNode3, std::move(Edge));

    if (Classifier.has_value()) {
      Classifier->addBackwardsEdgePartition(NewNode1, Node);
      Classifier->addBackwardsEdgePartition(NewNode1, NewNode2);
      Classifier->addBackwardsEdgePartition(NewNode2, NewNode3);
      Classifier->addBackwardsEdgePartition(Node, NewNode3);
    }

    Ranks[NewNode1] = Ranks.at(Node) - 1;
    Ranks[NewNode2] = Ranks.at(Node);
    Ranks[NewNode3] = Ranks.at(Node) + 1;
  }
}

template<RankingStrategy Strategy>
std::tuple<RankContainer, MaybeClassifier<Strategy>>
prepareGraph(InternalGraph &Graph, bool ShouldOmitClassification) {
  // Temporarily remove self-loops from the graph.
  auto SelfLoops = extractSelfLoops(Graph);

  // Temporarily reverse some of the edges so the graph doesn't contain loops.
  convertToDAG(Graph);

  // Use a robust node classification to speed the permutation selection up.
  MaybeClassifier<Strategy> Classifier;
  if (!ShouldOmitClassification)
    Classifier = NodeClassifier<Strategy>{};

  // Split long edges into one rank wide partitions.
  auto Ranks = partitionLongEdges(Graph, Classifier);

  // Split backwards facing edges created when partitioning the long edges up.
  partitionArtificialBackwardsEdges(Graph, Ranks, Classifier);

  // Split the backwards facing edges from the "external" graph into partitions.
  partitionOriginalBackwardsEdges(Graph, Ranks, Classifier);

  // Add the self-loops back in a partitioned form.
  partitionSelfLoops(Graph, Ranks, SelfLoops, Classifier);

  return { std::move(Ranks), std::move(Classifier) };
}

template<RankingStrategy Strategy>
using ResultTuple = std::tuple<RankContainer, MaybeClassifier<Strategy>>;

template ResultTuple<RankingStrategy::BreadthFirstSearch>
prepareGraph<RankingStrategy::BreadthFirstSearch>(InternalGraph &, bool);

template ResultTuple<RankingStrategy::DepthFirstSearch>
prepareGraph<RankingStrategy::DepthFirstSearch>(InternalGraph &, bool);

template ResultTuple<RankingStrategy::Topological>
prepareGraph<RankingStrategy::Topological>(InternalGraph &, bool);

template ResultTuple<RankingStrategy::DisjointDepthFirstSearch>
prepareGraph<RankingStrategy::DisjointDepthFirstSearch>(InternalGraph &, bool);
