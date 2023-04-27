#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Yield/Support/GraphLayout/Graphs.h"
#include "revng/Yield/Support/GraphLayout/SugiyamaStyle/InternalGraph.h"

namespace yield::layout::sugiyama {

/// Lists possible ranking strategies the layouter implements.
enum class RankingStrategy {
  BreadthFirstSearch,
  DepthFirstSearch,
  Topological,
  DisjointDepthFirstSearch
};

/// List graph orientation options the layouter implements.
enum class Orientation { LeftToRight, RightToLeft, TopToBottom, BottomToTop };

struct Configuration {
public:
  /// Specifies the way the layers of the graph are decided
  RankingStrategy Ranking;

  /// Specifies the orientation of the layout, (e.g. left-to-right):
  /// The layers with lower ranks are closer to the first direction (e.g. left)
  /// while those with higher ranks - to the second one (e.g. right).
  Orientation Orientation;

  /// Specifies whether orthogonal bending rules should be used, if not each
  /// edge is just two points: its start and end.
  bool UseOrthogonalBends;

  /// Specifies whether linear segment preservation should be done.
  ///
  /// This causes nodes any pair of nodes (A and B) such that A has only one
  /// successor - B and B has only one predecessor - A to be always be routed
  /// with a straight line (they both have the same horizontal position).
  bool PreserveLinearSegments;

  /// Specifies whether simple tree optimizations should be used.
  ///
  /// A simple tree expects every single node to only have one predecessor and
  /// to not have any back-edges.
  bool UseSimpleTreeOptimization;

  /// Specifies the "weight" of virtual nodes as opposed to the real ones.
  ///
  /// Virtual nodes represent long edges (both forwards and backwards facing).
  ///
  /// If weight is equal to 1, the placing of virtual nodes is identical to that
  /// of the real ones.
  ///
  /// If weight is bigger than 1 (e.g. 10), the virtual nodes are more likely
  /// to be placed to the right of the layout (the bigger the weight, the more
  /// noticeable its effect is).
  ///
  /// If weight is smaller than 1 (e.g. 0.1), the virtual nodes are more likely
  /// to be placed to the left of the layout (the smaller the weight, the more
  /// noticeable its effect is).
  ///
  /// \note: the layouts will be unnaturally skewed if this value is negative.
  float VirtualNodeWeight;

  /// Specifies the minimum possible distance between two nodes.
  layout::Dimension NodeMarginSize;

  /// Specifies the minimum possible distance between two edges.
  layout::Dimension EdgeMarginSize;
};

namespace detail {

bool computeImpl(InternalGraph &Internal, const Configuration &Configuration);

} // namespace detail

template<layout::HasLayoutableGraphTraits GraphType>
inline bool
computeInPlace(GraphType &&Graph, const Configuration &Configuration) {
  using IG = InternalGraph;
  auto [Internal, InputNodeLookup] = IG::make(std::forward<GraphType>(Graph));
  if (!detail::computeImpl(Internal, Configuration))
    return false;

  Internal.template exportInto<GraphType>(InputNodeLookup);
  return true;
}

/// A custom graph layering algorithm designed for pre-calculating majority of
/// the expensive stuff needed for graph rendering.
///
/// \tparam Node The type of the data attached to each graph node
/// \tparam Edge The type of the data attached to each graph edge
///
/// \param Graph An input graph
/// \param Configuration An object configuring the specifics of the layout
///
/// \return The laid out version of the graph corresponding to \ref Graph
template<typename Node, typename Edge = Empty>
inline std::optional<layout::OutputGraph<Node, Edge>>
compute(const layout::InputGraph<Node, Edge> &Graph,
        const Configuration &Configuration) {
  // TODO: rename into `compute` once `llvm::GraphTraits` has a concept support
  //       for checking whether it's defined for a given type or not.

  using OutputGraph = layout::OutputGraph<Node, Edge>;
  std::optional<OutputGraph> Result = layout::detail::convert(Graph);
  if (!computeInPlace(&Result.value(), Configuration))
    Result = std::nullopt;

  return Result;
}

} // namespace yield::layout::sugiyama
