#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "InternalGraph.h"
#include "NodeClassification.h"

/// Prepares the graph for further processing.
template<RankingStrategy Strategy>
std::tuple<InternalGraph, RankContainer, MaybeClassifier<Strategy>>
prepareGraph(ExternalGraph &Graph, bool OmitClassification);

/// Approximates an optimal permutation selection.
template<RankingStrategy Strategy>
LayerContainer selectPermutation(InternalGraph &Graph,
                                 RankContainer &Ranks,
                                 const MaybeClassifier<Strategy> &Classifier);

/// A simplified permutation selection to only be used with simple tree.
LayerContainer
selectSimpleTreePermutation(InternalGraph &Graph, RankContainer &Ranks);

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

/// Builds an empty linear segments map.
SegmentContainer emptyLinearSegments(InternalGraph &Graph);

/// "Levels up" a `LayerContainer` to a `LayoutContainer`.
LayoutContainer convertToLayout(const LayerContainer &Layers);

/// Calculates horizontal coordinates based on a finalized layout and segments.
void setHorizontalCoordinates(const LayerContainer &Layers,
                              const std::vector<NodeView> &Order,
                              const SegmentContainer &LinearSegments,
                              const LayoutContainer &Layout,
                              float MarginSize,
                              float VirtualNodeWeight = 0.1f);

/// Simplified horizontal coordinate calculation based on layers only.
void setStaticOffsetHorizontalCoordinates(const LayerContainer &Layers,
                                          float MarginSize);

/// Distributes "touching" edges across lanes to minimize the crossing count.
LaneContainer assignLanes(InternalGraph &Graph,
                          const SegmentContainer &LinearSegments,
                          const LayoutContainer &Layout);

/// Calculates vertical coordinates based on layer and lane data.
void setVerticalCoordinates(const LayerContainer &Layers,
                            const LaneContainer &Lanes,
                            float MarginSize,
                            float EdgeDistance);

/// Routes edges that form backwards facing corners. For their indication,
/// V-shaped structures were added to the graph when the backwards edges
/// were partitioned.
CornerContainer routeBackwardsCorners(InternalGraph &Graph,
                                      const RankContainer &Ranks,
                                      const LaneContainer &Lanes,
                                      float MarginSize,
                                      float EdgeDistance);

/// Consumes a DAG to produce the optimal routing order.
OrderedEdgeContainer orderEdges(InternalGraph &&Graph,
                                CornerContainer &&Prerouted,
                                const RankContainer &Ranks,
                                const LaneContainer &Lanes);

void route(const OrderedEdgeContainer &OrderedListOfEdges,
           float MarginSize,
           float EdgeDistance);

void routeWithStraightLines(const OrderedEdgeContainer &OrderedListOfEdges);

/// Computes the layout given a graph and the configuration.
///
/// \note: it only works with `MutableEdgeNode`s.
template<yield::layout::sugiyama::RankingStrategy RS>
bool computeInternal(ExternalGraph &Graph, const Configuration &Configuration) {
  static_assert(StrictSpecializationOfMutableEdgeNode<InternalNode>,
                "LayouterSugiyama requires mutable edge nodes.");

  // There's nothing to lay out in a graph without any nodes.
  if (Graph.size() == 0)
    return true;

  // Prepare the graph for the layouter: this converts `Graph` into
  // an internal graph and guaranties that it's has no loops (some of the
  // edges might have to be temporarily inverted to ensure this), a single
  // entry point (an extra node might have to be added) and that both
  // long edges and backwards facing edges are split up into into chunks
  // that span at most one layer at a time.
  bool ShouldClassify = !Configuration.UseSimpleTreeOptimization;
  auto [DAG, Ranks, Classified] = prepareGraph<RS>(Graph, !ShouldClassify);

  // Try to select an optimal node permutation per layer.
  // NOTE: since this is the part with the highest complexity, it needs extra
  // care for the layouter to perform well.
  // Maybe we should consider something more optimal instead of a simple hill
  // climbing algorithm.
  auto Layers = Configuration.UseSimpleTreeOptimization ?
                  selectSimpleTreePermutation(DAG, Ranks) :
                  selectPermutation<RS>(DAG, Ranks, *Classified);

  // Compute an augmented topological ordering of the nodes of the graph.
  auto Order = extractAugmentedTopologicalOrder(DAG, Layers);

  // Decide on which segments of the graph can be made linear, e.g. each edge
  // within the same linear segment is a straight line.
  SegmentContainer LinearSegments;
  if (Configuration.PreserveLinearSegments)
    LinearSegments = selectLinearSegments(DAG, Ranks, Layers, Order);
  else
    LinearSegments = emptyLinearSegments(DAG);

  // Finalize the logical positions for each of the nodes.
  const auto Final = convertToLayout(Layers);

  // Finalize the horizontal node positions.
  const auto &Margin = Configuration.NodeMarginSize;
  if (Configuration.UseSimpleTreeOptimization) {
    size_t MaximumNodeWidth = 0;
    for (auto *Node : Graph.nodes())
      if (Node->Size.W > MaximumNodeWidth)
        MaximumNodeWidth = Node->Size.W;
    setStaticOffsetHorizontalCoordinates(Layers, MaximumNodeWidth + Margin);
  } else {
    const auto &W = Configuration.VirtualNodeWeight;
    setHorizontalCoordinates(Layers, Order, LinearSegments, Final, Margin, W);
  }

  // Distribute edge lanes in a way that minimizes the number of crossings.
  auto Lanes = assignLanes(DAG, LinearSegments, Final);

  // Set the rest of the coordinates. Node layouting is complete after this.
  const auto &EdgeGap = Configuration.EdgeMarginSize;
  setVerticalCoordinates(Layers, Lanes, Margin, EdgeGap);

  // Route edges forming backwards facing corners.
  CornerContainer Prerouted;
  if (Configuration.UseOrthogonalBends)
    Prerouted = routeBackwardsCorners(DAG, Ranks, Lanes, Margin, EdgeGap);

  // Now that the corners are routed, the DAG representation is not needed
  // anymore, both the graph and the routed corners get consumed to construct
  // an ordered list of edges with all the information necessary for them
  // to get routed (see `OrderedEdgeContainer`).
  auto Edges = orderEdges(std::move(DAG), std::move(Prerouted), Ranks, Lanes);

  // Route the edges.
  if (Configuration.UseOrthogonalBends)
    route(Edges, Margin, EdgeGap);
  else
    routeWithStraightLines(Edges);

  return true;
}
