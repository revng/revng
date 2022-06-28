#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Yield/ControlFlow/Configuration.h"
#include "revng/Yield/Graph.h"

namespace yield::sugiyama {

enum class RankingStrategy {
  BreadthFirstSearch,
  DepthFirstSearch,
  Topological,
  DisjointDepthFirstSearch
};

enum class LayoutOrientation {
  LeftToRight,
  RightToLeft,
  TopToBottom,
  BottomToTop
};

struct Configuration {
public:
  RankingStrategy Ranking;
  LayoutOrientation Orientation;
  bool UseOrthogonalBends;

  bool PreserveLinearSegments;
  float VirtualNodeWeight;

  Graph::Dimension NodeMarginSize;
  Graph::Dimension EdgeMarginSize;
};

/// A custom graph layering algorithm designed for pre-calculating majority of
/// the expensive stuff needed for graph rendering.
bool layout(Graph &Graph, const Configuration &Configuration);

inline bool
layout(Graph &Graph,
       const cfg::Configuration &CFG,
       RankingStrategy Ranking = RankingStrategy::DisjointDepthFirstSearch,
       LayoutOrientation Orientation = LayoutOrientation::TopToBottom) {
  return layout(Graph,
                Configuration{
                  .Ranking = Ranking,
                  .Orientation = Orientation,
                  .UseOrthogonalBends = CFG.UseOrthogonalBends,
                  .PreserveLinearSegments = CFG.PreserveLinearSegments,
                  .VirtualNodeWeight = CFG.VirtualNodeWeight,
                  .NodeMarginSize = CFG.ExternalNodeMarginSize,
                  .EdgeMarginSize = CFG.EdgeMarginSize });
}

} // namespace yield::sugiyama
