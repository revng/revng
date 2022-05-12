/// \file EdgeRouting.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "Layout.h"

CornerContainer routeBackwardsCorners(InternalGraph &Graph,
                                      const RankContainer &Ranks,
                                      const LaneContainer &Lanes,
                                      float MarginSize,
                                      float EdgeDistance) {
  std::vector<DirectedEdgeView> CornerEdges;

  // To keep the hierarchy consistent, V-shapes were added using forward
  // direction. So that's what we're going to use to detect them.
  for (auto *From : Graph.nodes())
    for (auto [To, Label] : From->successor_edges())
      if (!From->isVirtual() != !To->isVirtual() && !Label->IsBackwards)
        CornerEdges.emplace_back(From, To, Label->Pointer, false);

  CornerContainer Corners;
  for (auto &Edge : CornerEdges) {
    auto LaneIndex = 0;
    auto Rank = std::min(Ranks.at(Edge.From), Ranks.at(Edge.To));
    if (Rank < Lanes.Horizontal.size()) {
      auto &CurrentLayerLanes = Lanes.Horizontal.at(Rank);
      auto Iterator = CurrentLayerLanes.find(Edge);
      if (Iterator != CurrentLayerLanes.end())
        LaneIndex = Iterator->second;
    }

    if (Edge.From->isVirtual() && !Edge.To->isVirtual()) {
      if (Edge.From->successorCount() != 2 || Edge.From->hasPredecessors())
        continue;

      // One side of the corner.
      auto *First = *Edge.From->successors().begin();

      // The other side.
      auto *Second = *std::next(Edge.From->successors().begin());

      // Make sure there are no self-loops, otherwise it's not a corner.
      if (First->Index == Edge.From->Index || Second->Index == Edge.From->Index)
        continue;

      auto ToUpperEdge = First->center().Y + First->size().H / 2;
      auto FromUpperEdge = Second->center().Y + Second->size().H / 2;
      Edge.From->center().X = (First->center().X + Second->center().X) / 2;
      Edge.From->center().Y = std::min(ToUpperEdge, FromUpperEdge) + MarginSize
                              + LaneIndex * EdgeDistance;

      auto &From = Edge.From;
      for (auto [To, Label] : From->successor_edges()) {
        auto FromTop = From->center().Y + From->size().H / 2;
        auto ToTop = To->center().Y + To->size().H / 2;

        if (Label->IsBackwards) {
          revng_assert(!Corners.contains({ To, From }));

          auto FromPoint = Point{ To->center().X, ToTop };
          auto CenterPoint = Point{ To->center().X, From->center().Y };
          auto ToPoint = Point{ From->center().X, FromTop };

          Corners.emplace(NodePair{ To, From },
                          Corner{ FromPoint, CenterPoint, ToPoint });
        } else {
          revng_assert(!Corners.contains({ From, To }));

          auto ToLane = To->center().X;
          if (auto It = Lanes.Entries.find(To); It != Lanes.Entries.end()) {
            auto View = DirectedEdgeView{ From, To, Label->Pointer, false };
            revng_assert(It->second.contains(View));

            auto EntryIndex = float(It->second.at(View));
            auto CenteredIndex = EntryIndex - float(It->second.size() - 1) / 2;

            auto ToLaneGap = EdgeDistance / 2;
            if (It->second.size() != 0) {
              auto AlternativeGap = To->size().W / 2 / It->second.size();
              if (AlternativeGap < ToLaneGap)
                ToLaneGap = AlternativeGap;
            }

            ToLane += ToLaneGap * CenteredIndex;
          }

          auto FromPoint = Point{ From->center().X, FromTop };
          auto CenterPoint = Point{ ToLane, From->center().Y };
          auto ToPoint = Point{ ToLane, ToTop };

          Corners.emplace(NodePair{ From, To },
                          Corner{ FromPoint, CenterPoint, ToPoint });
        }
      }
    } else {
      if (Edge.To->predecessorCount() != 2 || Edge.To->hasSuccessors())
        continue;

      // One side of the corner.
      auto *First = *Edge.To->predecessors().begin();

      // The other side.
      auto *Second = *std::next(Edge.To->predecessors().begin());

      // Make sure there are no self-loops, otherwise it's not a corner.
      if (First->Index == Edge.To->Index || Second->Index == Edge.To->Index)
        continue;

      Edge.To->center().X = (First->center().X + Second->center().X) / 2;
      Edge.To->center().Y += MarginSize + LaneIndex * EdgeDistance;

      auto &To = Edge.To;
      for (auto [From, Label] : To->predecessor_edges()) {
        auto FromBottom = From->center().Y - From->size().H / 2;
        auto ToBottom = To->center().Y - To->size().H / 2;

        if (Label->IsBackwards) {
          revng_assert(!Corners.contains({ To, From }));

          auto FromPoint = Point{ To->center().X, ToBottom };
          auto CenterPoint = Point{ From->center().X, To->center().Y };
          auto ToPoint = Point{ From->center().X, FromBottom };

          Corners.emplace(NodePair{ To, From },
                          Corner{ FromPoint, CenterPoint, ToPoint });
        } else {
          revng_assert(!Corners.contains({ From, To }));

          auto FromLane = From->center().X;
          if (auto It = Lanes.Exits.find(From); It != Lanes.Exits.end()) {
            auto View = DirectedEdgeView{ From, To, Label->Pointer, false };
            revng_assert(It->second.contains(View));

            auto ExitIndex = float(It->second.at(View));
            auto CenteredIndex = ExitIndex - float(It->second.size() - 1) / 2;

            auto FromLaneGap = EdgeDistance / 2;
            if (It->second.size() != 0) {
              auto AlternativeGap = From->size().W / 2 / It->second.size();
              if (AlternativeGap < FromLaneGap)
                FromLaneGap = AlternativeGap;
            }

            FromLane += FromLaneGap * CenteredIndex;
          }

          auto FromPoint = Point{ FromLane, FromBottom };
          auto CenterPoint = Point{ FromLane, To->center().Y };
          auto ToPoint = Point{ To->center().X, ToBottom };

          Corners.emplace(NodePair{ From, To },
                          Corner{ FromPoint, CenterPoint, ToPoint });
        }
      }
    }
  }

  return Corners;
}
