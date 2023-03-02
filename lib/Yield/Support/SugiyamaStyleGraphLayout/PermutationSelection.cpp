/// \file PermutationSelection.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>

#include "llvm/ADT/PostOrderIterator.h"

#include "Layout.h"

/// Converts given rankings to a layer container and updates ranks to remove
/// the layers that are not required for correct routing.
static LayerContainer
optimizeLayers(InternalGraph &Graph, RankContainer &Ranks) {
  LayerContainer Layers;
  for (auto &[Node, Rank] : Ranks) {
    if (Rank >= Layers.size())
      Layers.resize(Rank + 1);
    Layers[Rank].emplace_back(Node);
  }

  for (auto Iterator = Layers.begin(); Iterator != Layers.end();) {
    bool IsLayerRequired = false;
    for (auto Node : *Iterator) {
      // In order for a node to be easily removable, it shouldn't have
      // an external counterpart (meaning it was added when some of the edges
      // were partitioned) and shouldn't be necessary for backwards edge
      // routing (not a part of a V-shaped structure).
      //
      // Such nodes always have a single predecessor and a single successor.
      // Additionally, the ranks of those two neighbors have to be different.
      if (!Node->isVirtual()) {
        IsLayerRequired = true;
        break;
      }
      if ((Node->successorCount() != 1) || (Node->predecessorCount() != 1)) {
        IsLayerRequired = true;
        break;
      }

      auto SuccessorRank = Ranks.at(*Node->successors().begin());
      auto PredecessorRank = Ranks.at(*Node->predecessors().begin());
      if (SuccessorRank == PredecessorRank) {
        IsLayerRequired = true;
        break;
      }
    }

    if (!IsLayerRequired) {
      for (auto Node : *Iterator) {
        auto *Predecessor = *Node->predecessors().begin();
        auto *Successor = *Node->successors().begin();
        auto Label = std::move(*Node->predecessor_edges().begin()->Label);
        Predecessor->addSuccessor(Successor, std::move(Label));
        Ranks.erase(Node);
        Graph.removeNode(Node);
      }

      Iterator = Layers.erase(Iterator);
    } else {
      ++Iterator;
    }
  }

  // Update ranks
  for (size_t Index = 0; Index < Layers.size(); ++Index)
    for (auto Node : Layers[Index])
      Ranks.at(Node) = Index;

  revng_assert(Ranks.size() == Graph.size());
  return Layers;
}

/// Counts the total number of nodes within a `Layers` container.
static size_t countNodes(const LayerContainer &Layers) {
  size_t Counter = 0;
  for (auto &Layer : Layers)
    Counter += Layer.size();
  return Counter;
}

struct CrossingCalculator {
public:
  const LayerContainer &Layers; // A view onto the layered nodes.
  const RankContainer &Ranks; // A view onto node rankings.
  RankContainer &Permutation; // A view onto a current permutation.

private:
  /// Counts the number of edge crossings given two nodes and
  /// a map that represents which nodes in an adjacent layer
  /// are connected to one of the given nodes.
  static Rank countImpl(NodeView KNode,
                        NodeView LNode,
                        const std::map<Rank, bool> &SortedLayer,
                        const RankContainer &Permutation) {
    Rank CrossingCount = 0;
    if (SortedLayer.size() > 0) {
      bool KSide = SortedLayer.begin()->second;
      bool KLeft = Permutation.at(KNode) < Permutation.at(LNode);
      int64_t PreviousSegmentSize = 0;
      int64_t CurrentSegmentSize = 0;

      for (auto &[Position, Side] : SortedLayer) {
        if (Side == KSide) {
          CurrentSegmentSize += 1;
        } else {
          if (KSide && KLeft)
            CrossingCount += PreviousSegmentSize * CurrentSegmentSize;

          PreviousSegmentSize = CurrentSegmentSize;
          CurrentSegmentSize = 1;
          KSide = Side;
        }
      }

      if (KSide && KLeft)
        CrossingCount += PreviousSegmentSize * CurrentSegmentSize;
    }

    return CrossingCount;
  }

public:
  // Counts the crossings.
  Rank countCrossings(Rank CurrentRank, NodeView KNode, NodeView LNode) {
    revng_assert(CurrentRank < Layers.size());

    Rank CrossingCount = 0;

    if (CurrentRank != 0) {
      std::map<Rank, bool> SortedLayer;
      for (auto *Predecessor : KNode->predecessors())
        if (Ranks.at(Predecessor) == CurrentRank - 1)
          SortedLayer[Permutation.at(Predecessor)] = true;

      for (auto *Predecessor : LNode->predecessors())
        if (Ranks.at(Predecessor) == CurrentRank - 1)
          SortedLayer[Permutation.at(Predecessor)] = false;

      CrossingCount += countImpl(KNode, LNode, SortedLayer, Permutation);
    }

    if (CurrentRank != Layers.size() - 1) {
      std::map<Rank, bool> SortedLayer;
      for (auto *Successor : KNode->successors())
        if (Ranks.at(Successor) == CurrentRank + 1)
          SortedLayer[Permutation.at(Successor)] = true;

      for (auto *Successor : LNode->successors())
        if (Ranks.at(Successor) == CurrentRank + 1)
          SortedLayer[Permutation.at(Successor)] = false;

      CrossingCount += countImpl(KNode, LNode, SortedLayer, Permutation);
    }

    return CrossingCount;
  }

  /// Computes the difference in the crossing count
  /// based on the node positions (e.g. how much better/worse the crossing
  /// count becomes if a permutation were to be applied).
  RankDelta computeDelta(Rank CurrentRank, NodeView KNode, NodeView LNode) {
    auto KIterator = Permutation.find(KNode);
    auto LIterator = Permutation.find(LNode);
    revng_assert(KIterator != Permutation.end());
    revng_assert(LIterator != Permutation.end());

    auto OriginalCrossingCount = countCrossings(CurrentRank, KNode, LNode);
    std::swap(KIterator->second, LIterator->second);
    auto NewCrossingCount = countCrossings(CurrentRank, KNode, LNode);
    std::swap(KIterator->second, LIterator->second);

    return RankDelta(NewCrossingCount) - RankDelta(OriginalCrossingCount);
  }
};

/// Minimizes crossing count using a simple hill climbing algorithm.
/// The function can be sped up by providing an initial permutation found
/// using other techniques.
/// A function used for horizontal node segmentation can also be specified.
template<typename ClusterType>
LayerContainer minimizeCrossingCount(const RankContainer &Ranks,
                                     const ClusterType &Cluster,
                                     LayerContainer &&Layers) {
  revng_assert(countNodes(Layers) == Ranks.size());

  // In principle we'd like to compute the iteration count from some features in
  // the graph (number of nodes, number of edges, number of layers, number of
  // nodes in each layer, and so on...) so that, no matter how complex the
  // graph is, it's will always possible to terminate in an amount of time
  // that is guaranteed not to exceed an arbitrary hard limit.
  //
  // The problem is that we do not know how does the complexity of the graph
  // translate to minimization time, hence we run some experiments and select
  // a `ReferenceComplexity`. This number is an `IterationComplexity` of
  // a hand-picked graph that's moderately fast to lay out.
  constexpr size_t ReferenceComplexity = 70000;

  // To compare the current graph to the reference, we need to calculate
  // the `IterationComplexity` for the current graph:
  //
  // IterationComplexity = \sum_{i=0}^{Layers.size()}Layers.at(i).size()^2
  size_t IterationComplexity = 0;

  RankContainer Permutation;
  for (auto &Layer : Layers) {
    size_t LayerSize = Layer.size();
    for (size_t I = 0; I < LayerSize; I++)
      Permutation[Layer[I]] = I;
    IterationComplexity += LayerSize * LayerSize;
  }

  // The idea is that we cannot afford to do more than a single iteration
  // on any graph that's more complex than the reference one. On the other hand,
  // it's practical to allow graphs with complexity beneath that of the
  // reference to do multiple crossing minimization iterations.
  // The exact number depends on the ratio between the graph iteration
  // complexity to the reference iteration complexity.
  //
  // The reasoning is more or less like this: if the `IterationComplexity` is
  // small, each iteration is cheap and we can afford to do many iterations.
  // On the other hand, if the graph is large, the ratio rapidly goes to zero
  // and the iteration count drop to just a single one.
  //
  // TODO: we likely want to put a hardcoded cap on the max number of iterations
  size_t IterationCount = 1 + ReferenceComplexity / IterationComplexity;

  auto Comparator = [&Cluster, &Permutation](NodeView A, NodeView B) {
    if (Cluster(A) == Cluster(B))
      return Permutation.at(A) < Permutation.at(B);
    else
      return Cluster(A) < Cluster(B);
  };

  CrossingCalculator Calculator{ Layers, Ranks, Permutation };
  for (size_t Iteration = 0; Iteration < IterationCount; ++Iteration) {
    for (size_t Index = 0; Index < Layers.size(); ++Index) {
      if (size_t CurrentLayerSize = Layers[Index].size(); CurrentLayerSize) {

        // Minimize WRT of the previous layer
        // This can be expensive so we limit the number of times we repeat it.
        std::sort(Layers[Index].begin(), Layers[Index].end(), Comparator);
        for (size_t NodeIndex = 0; NodeIndex < CurrentLayerSize; ++NodeIndex)
          Permutation[Layers[Index][NodeIndex]] = NodeIndex;

        for (size_t NodeIndex = 0; NodeIndex < CurrentLayerSize; ++NodeIndex) {
          RankDelta ChoosenDelta = 0;
          std::pair<Rank, Rank> ChoosenNodes;

          for (size_t K = 0; K < CurrentLayerSize; ++K) {
            for (size_t L = K + 1; L < CurrentLayerSize; ++L) {
              auto KNode = Layers[Index][K];
              auto LNode = Layers[Index][L];
              auto Delta = Calculator.computeDelta(Index, KNode, LNode);
              if (Delta < ChoosenDelta) {
                ChoosenDelta = Delta;
                ChoosenNodes = { K, L };
              }
            }
          }

          if (ChoosenDelta == 0)
            break;

          auto KNode = Layers[Index][ChoosenNodes.first];
          auto LNode = Layers[Index][ChoosenNodes.second];
          std::swap(Permutation.at(KNode), Permutation.at(LNode));
        }

        std::sort(Layers[Index].begin(), Layers[Index].end(), Comparator);
        for (size_t NodeIndex = 0; NodeIndex < CurrentLayerSize; ++NodeIndex)
          Permutation[Layers[Index][NodeIndex]] = NodeIndex;
      }
    }
  }

  LayerContainer Result;
  for (auto &[Node, Rank] : Ranks) {
    if (Rank >= Result.size())
      Result.resize(Rank + 1);
    Result[Rank].emplace_back(Node);
  }
  for (auto &Layer : Result)
    std::sort(Layer.begin(),
              Layer.end(),
              [&Permutation](const auto &LHS, const auto &RHS) {
                return Permutation.at(LHS) < Permutation.at(RHS);
              });
  return Result;
}

template<bool PreOrPost, typename ClusterType>
class BarycentricComparator {
public:
  BarycentricComparator(const RankContainer &Ranks,
                        const RankContainer &Positions,
                        const LayerContainer &Layers,
                        const ClusterType &Cluster) :
    Ranks(Ranks), Positions(Positions), Layers(Layers), Cluster(Cluster) {}

  bool operator()(NodeView LHS, NodeView RHS) {
    if (int A = Cluster(LHS), B = Cluster(RHS); A == B) {
      auto BarycenterA = get(LHS), BarycenterB = get(RHS);
      if (std::isnan(BarycenterA) || std::isnan(BarycenterB))
        return LHS->Index < RHS->Index;
      else
        return BarycenterA < BarycenterB;
    } else {
      return A < B;
    }
  }

protected:
  double get(NodeView Node) {
    auto Iterator = Barycenters.lower_bound(Node);
    if (Iterator == Barycenters.end() || Node < Iterator->first)
      Iterator = Barycenters.insert(Iterator, { Node, compute(Node) });
    return Iterator->second;
  }

  double computeImpl(NodeView Node, auto NeighborContainerView) {
    if (NeighborContainerView.empty())
      return std::numeric_limits<double>::quiet_NaN();

    double Accumulator = 0;
    size_t Counter = 0;

    double CurrentLayerSize = Layers.at(Ranks.at(Node)).size();
    for (auto *Neighbor : NeighborContainerView) {
      if (auto It = Positions.find(Neighbor); It != Positions.end()) {
        auto NeighborLayerSize = Layers.at(Ranks.at(Neighbor)).size();
        Accumulator += (CurrentLayerSize / NeighborLayerSize) * It->second;
        ++Counter;
      }
    }

    if (Counter == 0)
      return std::numeric_limits<double>::quiet_NaN();
    return Accumulator / Counter;
  }

  double compute(NodeView Node) {
    if constexpr (PreOrPost)
      return computeImpl(Node, Node->predecessors());
    else
      return computeImpl(Node, Node->successors());
  }

private:
  std::map<NodeView, double> Barycenters;
  const RankContainer &Ranks;
  const RankContainer &Positions;
  const LayerContainer &Layers;
  const ClusterType &Cluster;
};

/// Sorts the permutations using the barycentric sorting described in
/// "A fast heuristic for hierarchical Manhattan layout" by G. Sander (2005)
template<typename ClusterType>
LayerContainer sortNodes(const RankContainer &Ranks,
                         const size_t IterationCount,
                         const ClusterType &Cluster,
                         LayerContainer &&Layers) {
  revng_assert(countNodes(Layers) == Ranks.size());

  RankContainer Positions;
  for (size_t Iteration = 0; Iteration < IterationCount; Iteration++) {
    for (size_t Index = 0; Index < Layers.size(); ++Index) {
      for (size_t Counter = 0; auto Node : Layers[Index])
        Positions[Node] = Counter++;

      using PBC = BarycentricComparator<true, decltype(Cluster)>;
      PBC Comparator(Ranks, Positions, Layers, Cluster);
      std::sort(Layers[Index].begin(), Layers[Index].end(), Comparator);

      for (size_t Counter = 0; auto Node : Layers[Index])
        Positions[Node] = Counter++;
    }
    for (size_t Index = Layers.size() - 1; Index != size_t(-1); --Index) {
      for (size_t Counter = 0; auto Node : Layers[Index])
        Positions[Node] = Counter++;

      using PBC = BarycentricComparator<false, decltype(Cluster)>;
      PBC Comparator(Ranks, Positions, Layers, Cluster);
      std::sort(Layers[Index].begin(), Layers[Index].end(), Comparator);

      for (size_t Counter = 0; auto Node : Layers[Index])
        Positions[Node] = Counter++;
    }
  }
  return std::move(Layers);
}

template<RankingStrategy Strategy>
LayerContainer selectPermutation(InternalGraph &Graph,
                                 RankContainer &Ranks,
                                 const MaybeClassifier<Strategy> &Classifier) {
  revng_assert(Classifier.has_value());

  // Build a layer container based on a given ranking, then remove layers
  // without any original nodes or nodes that are required for correct
  // backwards edge routing. Update ranks accordingly.
  auto InitialLayers = optimizeLayers(Graph, Ranks);

  auto MinimalCrossingLayers = minimizeCrossingCount(Ranks,
                                                     *Classifier,
                                                     std::move(InitialLayers));

  // Iteration counts are chosen arbitrarily. If the computation time was not
  // an issue, we could keep iterating until convergence, but since it's not
  // the case, we have to choose a stopping point.
  //
  // The iteration count logarithmically depends on the layer number.
  size_t Iterations = std::log2(InitialLayers.size());

  if constexpr (Strategy == RankingStrategy::BreadthFirstSearch) {
    // \todo: Since layers are wide, the inner loops inside both crossing
    // minimization and node sorting are more costly on average.
    // Maybe inner loop orders should be randomized.
  }

  auto SortedLayers = sortNodes(Ranks,
                                Iterations * 3 + 10,
                                *Classifier,
                                std::move(MinimalCrossingLayers));

  return SortedLayers;
}

constexpr auto BFSRS = RankingStrategy::BreadthFirstSearch;
constexpr auto DFSRS = RankingStrategy::DepthFirstSearch;
constexpr auto TRS = RankingStrategy::Topological;
constexpr auto DDFSRS = RankingStrategy::DisjointDepthFirstSearch;

template LayerContainer
selectPermutation<BFSRS>(InternalGraph &Graph,
                         RankContainer &Ranks,
                         const MaybeClassifier<BFSRS> &Classifier);

template LayerContainer
selectPermutation<DFSRS>(InternalGraph &Graph,
                         RankContainer &Ranks,
                         const MaybeClassifier<DFSRS> &Classifier);

template LayerContainer
selectPermutation<TRS>(InternalGraph &Graph,
                       RankContainer &Ranks,
                       const MaybeClassifier<TRS> &Classifier);

template LayerContainer
selectPermutation<DDFSRS>(InternalGraph &Graph,
                          RankContainer &Ranks,
                          const MaybeClassifier<DDFSRS> &Classifier);

static std::unordered_map<NodeView, size_t> rankSubtrees(InternalGraph &Graph) {
  std::unordered_map<NodeView, size_t> Result;

  for (auto *Node : Graph.nodes()) {
    if (!Node->hasPredecessors()) {
      for (auto *CurrentNode : llvm::post_order(Node)) {
        size_t CurrentRank = 1;
        for (auto *Successor : CurrentNode->successors()) {
          auto Iterator = Result.find(Successor);
          revng_assert(Iterator != Result.end());
          CurrentRank += Iterator->second;
        }
        auto [_, Success] = Result.try_emplace(CurrentNode, CurrentRank);
        revng_assert(Success);
      }
    }
  }

  return Result;
}

LayerContainer
selectSimpleTreePermutation(InternalGraph &Graph, RankContainer &Ranks) {
  // Build a layer container based on a given ranking, then remove layers
  // that can be discarded without losing any improtant information, for example
  // layers only containing virtual nodes added when splitting long edges.
  // Update all the ranks so that the difference between the ranks of two layers
  // next to each other is always equal to one.
  auto InitialLayers = optimizeLayers(Graph, Ranks);

  // Build internal ranking lookup table based on subtree sizes.
  auto SubtreeLookup = rankSubtrees(Graph);

  // Define node rank comparator
  std::unordered_map<NodeView, Rank> LastLayerLookup;
  auto PredecessorRankComparator =
    [&SubtreeLookup, &LastLayerLookup](const auto &LHS, const auto RHS) {
      revng_assert(LHS->predecessorCount() <= 1);
      revng_assert(RHS->predecessorCount() <= 1);

      if (!LastLayerLookup.empty()) {
        auto LHSRank = std::numeric_limits<Rank>::max();
        auto RHSRank = std::numeric_limits<Rank>::max();

        if (LHS->hasPredecessors()) {
          auto LHSIterator = LastLayerLookup.find(*LHS->predecessors().begin());
          revng_assert(LHSIterator != LastLayerLookup.end());
          LHSRank = LHSIterator->second;
        }
        if (RHS->hasPredecessors()) {
          auto RHSIterator = LastLayerLookup.find(*RHS->predecessors().begin());
          revng_assert(RHSIterator != LastLayerLookup.end());
          RHSRank = RHSIterator->second;
        }

        if (LHSRank != RHSRank)
          return LHSRank < RHSRank;
      }

      auto LHSSubtreeSizeIterator = SubtreeLookup.find(LHS);
      revng_assert(LHSSubtreeSizeIterator != SubtreeLookup.end());
      auto RHSSubtreeSizeIterator = SubtreeLookup.find(RHS);
      revng_assert(RHSSubtreeSizeIterator != SubtreeLookup.end());

      return LHSSubtreeSizeIterator->second > RHSSubtreeSizeIterator->second;
    };

  for (auto &Layer : InitialLayers) {
    // Sort the layer.
    llvm::sort(Layer, PredecessorRankComparator);

    // Update the last layer lookup table.
    LastLayerLookup.clear();
    for (size_t Index = 0; Index < Layer.size(); ++Index) {
      auto [_, Success] = LastLayerLookup.try_emplace(Layer[Index], Index);
      revng_assert(Success);
    }
  }

  return InitialLayers;
}
