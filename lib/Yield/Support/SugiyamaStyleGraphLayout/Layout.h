#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "InternalGraph.h"
#include "NodeClassification.h"

/// Prepares the graph for futher processing.
template<RankingStrategy Strategy>
std::tuple<InternalGraph, RankContainer, NodeClassifier<Strategy>>
prepareGraph(ExternalGraph &Graph);

/// Approximates an optimal permutation selection.
template<RankingStrategy Strategy>
LayerContainer selectPermutation(InternalGraph &Graph,
                                 RankContainer &Ranks,
                                 const NodeClassifier<Strategy> &Classifier);

/// Topologically orders nodes of an augmented graph generated based on a
/// layered version of the graph.
std::vector<NodeView>
extractAugmentedTopologicalOrder(InternalGraph &Graph,
                                 const LayerContainer &Layers);

/// Looks for the linear segments and ensures an optimal combination of them
/// is selected. It uses an algorithm from the Sander's paper.
/// The worst case complexity is O(N^2) in the cases where jump table is huge,
/// but the common case is very far from that because normally both
/// entry and exit edge count is low (intuitively, our layouts are tall rather
/// than wide).
///
/// \note: it's probably a good idea to think about loosening the dependence
/// on tall graph layouts since we will want to also lay more generic graphs
/// out.
SegmentContainer selectLinearSegments(InternalGraph &Graph,
                                      const RankContainer &Ranks,
                                      const LayerContainer &Layers,
                                      const std::vector<NodeView> &Order);

/// "Levels up" a `LayerContainer` to a `LayoutContainer`.
LayoutContainer convertToLayout(const LayerContainer &Layers);

/// Calculates horizontal coordinates based on a finalized layout and segments.
void setHorizontalCoordinates(const LayerContainer &Layers,
                              const std::vector<NodeView> &Order,
                              const SegmentContainer &LinearSegments,
                              const LayoutContainer &Layout,
                              float MarginSize);

/// Computes the layout given a graph and the configuration.
///
/// \note: it only works with `MutableEdgeNode`s.
template<yield::sugiyama::RankingStrategy RS>
inline bool calculateSugiyamaLayout(ExternalGraph &Graph,
                                    const Configuration &Configuration) {
  static_assert(IsMutableEdgeNode<InternalNode>,
                "LayouterSugiyama requires mutable edge nodes.");

  // Prepare the graph for the layouter: this converts `Graph` into
  // an internal graph and guaranties that it's has no loops (some of the
  // edges might have to be temporarily inverted to ensure this), a single
  // entry point (an extra node might have to be added) and that both
  // long edges and backwards facing edges are split up into into chunks
  // that span at most one layer at a time.
  auto [DAG, Ranks, Classified] = prepareGraph<RS>(Graph);

  // Try to select an optimal node permutation per layer.
  // \note: since this is the part with the highest complexity, it needs extra
  // care for the layouter to perform well.
  // \suggestion: Maybe we should consider something more optimal instead of
  // a simple hill climbing algorithm.
  auto Layers = selectPermutation<RS>(DAG, Ranks, Classified);

  // Compute an augmented topological ordering of the nodes of the graph.
  auto Order = extractAugmentedTopologicalOrder(DAG, Layers);

  // Decide on which segments of the graph can be made linear, e.g. each edge
  // within the same linear segment is a straight line.
  auto LinearSegments = selectLinearSegments(DAG, Ranks, Layers, Order);

  // Finalize the logical positions for each of the nodes.
  const auto FinalLayout = convertToLayout(Layers);

  // Finalize the horizontal node positions.
  const auto &Margin = Configuration.NodeMarginSize;
  setHorizontalCoordinates(Layers, Order, LinearSegments, FinalLayout, Margin);

  return true;
}
