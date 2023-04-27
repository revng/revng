#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <unordered_set>

#include "Helpers.h"

namespace detail {
// Contains helper classes for `NodeClassifier`.

template<RankingStrategy RS>
class NodeClassifierStorage;

template<>
class NodeClassifierStorage<RankingStrategy::BreadthFirstSearch> {
protected:
  std::unordered_set<NodeView> BackwardsEdgeNodes;
};

template<>
class NodeClassifierStorage<RankingStrategy::DepthFirstSearch>
  : public NodeClassifierStorage<RankingStrategy::BreadthFirstSearch> {
protected:
  std::unordered_set<NodeView> LongEdgeNodes;
};

template<>
class NodeClassifierStorage<RankingStrategy::Topological>
  : public NodeClassifierStorage<RankingStrategy::DepthFirstSearch> {};

template<>
class NodeClassifierStorage<RankingStrategy::DisjointDepthFirstSearch>
  : public NodeClassifierStorage<RankingStrategy::BreadthFirstSearch> {};

} // namespace detail

/// It is used to classify the nodes by selecting the right cluster depending
/// on its neighbors. This helps to make routes with edges routed across
/// multiple "virtual" nodes that require less bends.
template<RankingStrategy RS>
class NodeClassifier : public detail::NodeClassifierStorage<RS> {
  using Storage = detail::NodeClassifierStorage<RS>;

  enum Cluster { Left = 0, Middle = 1, Right = 2 };

public:
  void addBackwardsEdgePartition(NodeView LHS, NodeView RHS) {
    Storage::BackwardsEdgeNodes.emplace(LHS);
    Storage::BackwardsEdgeNodes.emplace(RHS);
  }

  void addLongEdgePartition(NodeView LHS, NodeView RHS) {
    if constexpr (RS == RankingStrategy::DepthFirstSearch
                  || RS == RankingStrategy::Topological) {
      Storage::LongEdgeNodes.emplace(LHS);
      Storage::LongEdgeNodes.emplace(RHS);
    }
  }

  size_t operator()(NodeView Node) const {
    if constexpr (RS == RankingStrategy::DepthFirstSearch
                  || RS == RankingStrategy::Topological)
      if (!Storage::LongEdgeNodes.contains(Node))
        return !Storage::BackwardsEdgeNodes.contains(Node) ? Middle : Right;
      else
        return Left;
    else
      return !Storage::BackwardsEdgeNodes.contains(Node) ? Left : Right;
  }
};

template<RankingStrategy Strategy>
using MaybeClassifier = std::optional<NodeClassifier<Strategy>>;
