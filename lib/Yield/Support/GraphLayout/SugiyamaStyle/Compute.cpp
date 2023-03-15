/// \file Compute.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Yield/Support/GraphLayout/SugiyamaStyle/Compute.h"

#include "Layout.h"

bool yield::sugiyama::layout(Graph &Graph, const Configuration &Configuration) {
  using RS = yield::sugiyama::RankingStrategy;

  if (Configuration.Orientation == LayoutOrientation::LeftToRight
      || Configuration.Orientation == LayoutOrientation::RightToLeft) {
    for (auto *Node : Graph.nodes())
      std::swap(Node->Size.W, Node->Size.H);
  }

  bool Res = false;
  switch (Configuration.Ranking) {
  case RS::BreadthFirstSearch:
    Res = calculateSugiyamaLayout<RS::BreadthFirstSearch>(Graph, Configuration);
    break;
  case RS::DepthFirstSearch:
    Res = calculateSugiyamaLayout<RS::DepthFirstSearch>(Graph, Configuration);
    break;
  case RS::Topological:
    Res = calculateSugiyamaLayout<RS::Topological>(Graph, Configuration);
    break;
  case RS::DisjointDepthFirstSearch:
    Res = calculateSugiyamaLayout<RS::DisjointDepthFirstSearch>(Graph,
                                                                Configuration);
    break;
  default:
    revng_abort("Unknown ranking strategy");
  }

  if (Res == false)
    return Res;

  if (Configuration.Orientation == LayoutOrientation::LeftToRight
      || Configuration.Orientation == LayoutOrientation::RightToLeft) {
    for (auto *Node : Graph.nodes()) {
      std::swap(Node->Size.W, Node->Size.H);
      std::swap(Node->Center.X, Node->Center.Y);

      for (auto [_, Edge] : Node->successor_edges())
        for (auto &[X, Y] : Edge->Path)
          std::swap(X, Y);
    }
  }

  if (Configuration.Orientation == LayoutOrientation::BottomToTop) {
    for (auto *Node : Graph.nodes()) {
      Node->Center.Y = -Node->Center.Y;

      for (auto [_, Edge] : Node->successor_edges())
        for (auto &[X, Y] : Edge->Path)
          Y = -Y;
    }
  } else if (Configuration.Orientation == LayoutOrientation::LeftToRight) {
    for (auto *Node : Graph.nodes()) {
      Node->Center.X = -Node->Center.X;

      for (auto [_, Edge] : Node->successor_edges())
        for (auto &[X, Y] : Edge->Path)
          X = -X;
    }
  }

  return Res;
}
