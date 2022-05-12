/// \file SugiyamaStyleGraphLayout.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Yield/Support/SugiyamaStyleGraphLayout.h"

#include "SugiyamaStyleGraphLayout/Layout.h"

bool yield::sugiyama::layout(Graph &Graph, const Configuration &Configuration) {
  using RS = yield::sugiyama::RankingStrategy;

  switch (Configuration.Ranking) {
  case RS::BreadthFirstSearch:
    return calculateSugiyamaLayout<RS::BreadthFirstSearch>(Graph,
                                                           Configuration);
  case RS::DepthFirstSearch:
    return calculateSugiyamaLayout<RS::DepthFirstSearch>(Graph, Configuration);
  case RS::Topological:
    return calculateSugiyamaLayout<RS::Topological>(Graph, Configuration);
  case RS::DisjointDepthFirstSearch:
    return calculateSugiyamaLayout<RS::DisjointDepthFirstSearch>(Graph,
                                                                 Configuration);
  default:
    revng_abort("Unknown ranking strategy");
  }
}
