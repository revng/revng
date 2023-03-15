/// \file Compute.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Yield/Support/GraphLayout/SugiyamaStyle/Compute.h"

#include "InternalCompute.h"

namespace sugiyama = yield::layout::sugiyama;
bool sugiyama::compute(yield::Graph &Graph,
                       const Configuration &Configuration) {
  using RS = sugiyama::RankingStrategy;

  if (Configuration.Orientation == sugiyama::Orientation::LeftToRight
      || Configuration.Orientation == sugiyama::Orientation::RightToLeft) {
    for (auto *Node : Graph.nodes())
      std::swap(Node->Size.W, Node->Size.H);
  }

  bool Res = false;
  switch (Configuration.Ranking) {
  case RS::BreadthFirstSearch:
    Res = computeInternal<RS::BreadthFirstSearch>(Graph, Configuration);
    break;
  case RS::DepthFirstSearch:
    Res = computeInternal<RS::DepthFirstSearch>(Graph, Configuration);
    break;
  case RS::Topological:
    Res = computeInternal<RS::Topological>(Graph, Configuration);
    break;
  case RS::DisjointDepthFirstSearch:
    Res = computeInternal<RS::DisjointDepthFirstSearch>(Graph, Configuration);
    break;
  default:
    revng_abort("Unknown ranking strategy");
  }

  if (Res == false)
    return Res;

  if (Configuration.Orientation == sugiyama::Orientation::LeftToRight
      || Configuration.Orientation == sugiyama::Orientation::RightToLeft) {
    for (auto *Node : Graph.nodes()) {
      std::swap(Node->Size.W, Node->Size.H);
      std::swap(Node->Center.X, Node->Center.Y);

      for (auto [_, Edge] : Node->successor_edges())
        for (auto &[X, Y] : Edge->Path)
          std::swap(X, Y);
    }
  }

  if (Configuration.Orientation == sugiyama::Orientation::BottomToTop) {
    for (auto *Node : Graph.nodes()) {
      Node->Center.Y = -Node->Center.Y;

      for (auto [_, Edge] : Node->successor_edges())
        for (auto &[X, Y] : Edge->Path)
          Y = -Y;
    }
  } else if (Configuration.Orientation == sugiyama::Orientation::LeftToRight) {
    for (auto *Node : Graph.nodes()) {
      Node->Center.X = -Node->Center.X;

      for (auto [_, Edge] : Node->successor_edges())
        for (auto &[X, Y] : Edge->Path)
          X = -X;
    }
  }

  return Res;
}
