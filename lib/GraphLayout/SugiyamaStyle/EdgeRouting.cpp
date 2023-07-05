/// \file EdgeRouting.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"

#include "InternalCompute.h"

CornerContainer routeBackwardsCorners(InternalGraph &Graph,
                                      const RankContainer &Ranks,
                                      const LaneContainer &Lanes,
                                      float MarginSize,
                                      float EdgeDistance) {
  std::vector<EdgeView> CornerEdges;

  // To keep the hierarchy consistent, V-shapes were added using forward
  // direction. So that's what we're going to use to detect them.
  for (auto *From : Graph.nodes())
    for (auto [To, Label] : From->successor_edges())
      if (!From->IsVirtual != !To->IsVirtual && !Label->IsBackwards)
        CornerEdges.emplace_back(From, To, *Label);

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

    if (Edge.From->IsVirtual && !Edge.To->IsVirtual) {
      if (Edge.From->successorCount() != 2 || Edge.From->hasPredecessors())
        continue;

      // One side of the corner.
      auto *First = *Edge.From->successors().begin();

      // The other side.
      auto *Second = *std::next(Edge.From->successors().begin());

      // Make sure there are no self-loops, otherwise it's not a corner.
      if (First->index() == Edge.From->index()
          || Second->index() == Edge.From->index())
        continue;

      auto ToUpperEdge = First->Center.Y + First->Size.H / 2;
      auto FromUpperEdge = Second->Center.Y + Second->Size.H / 2;
      Edge.From->Center.X = (First->Center.X + Second->Center.X) / 2;
      Edge.From->Center.Y = std::min(ToUpperEdge, FromUpperEdge) + MarginSize
                            + LaneIndex * EdgeDistance;

      auto &From = Edge.From;
      for (auto [To, Label] : From->successor_edges()) {
        auto FromTop = From->Center.Y + From->Size.H / 2;
        auto ToTop = To->Center.Y + To->Size.H / 2;

        if (Label->IsBackwards) {
          revng_assert(!Corners.contains({ To, From }));

          auto FromPoint = Point{ To->Center.X, ToTop };
          auto CenterPoint = Point{ To->Center.X, From->Center.Y };
          auto ToPoint = Point{ From->Center.X, FromTop };

          Corners.emplace(NodePair{ To, From },
                          Corner{ FromPoint, CenterPoint, ToPoint });
        } else {
          revng_assert(!Corners.contains({ From, To }));

          auto ToLane = To->Center.X;
          if (auto It = Lanes.Entries.find(To); It != Lanes.Entries.end()) {
            EdgeDestinationView View(From, *Label);
            revng_assert(It->second.contains(View));

            auto EntryIndex = float(It->second.at(View));
            auto CenteredIndex = EntryIndex - float(It->second.size() - 1) / 2;

            auto ToLaneGap = EdgeDistance / 2;
            if (It->second.size() != 0) {
              auto AlternativeGap = To->Size.W / 2 / It->second.size();
              if (AlternativeGap < ToLaneGap)
                ToLaneGap = AlternativeGap;
            }

            ToLane += ToLaneGap * CenteredIndex;
          }

          auto FromPoint = Point{ From->Center.X, FromTop };
          auto CenterPoint = Point{ ToLane, From->Center.Y };
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
      if (First->index() == Edge.To->index()
          || Second->index() == Edge.To->index())
        continue;

      Edge.To->Center.X = (First->Center.X + Second->Center.X) / 2;
      Edge.To->Center.Y += MarginSize + LaneIndex * EdgeDistance;

      auto &To = Edge.To;
      for (auto [From, Label] : To->predecessor_edges()) {
        auto FromBottom = From->Center.Y - From->Size.H / 2;
        auto ToBottom = To->Center.Y - To->Size.H / 2;

        if (Label->IsBackwards) {
          revng_assert(!Corners.contains({ To, From }));

          auto FromPoint = Point{ To->Center.X, ToBottom };
          auto CenterPoint = Point{ From->Center.X, To->Center.Y };
          auto ToPoint = Point{ From->Center.X, FromBottom };

          Corners.emplace(NodePair{ To, From },
                          Corner{ FromPoint, CenterPoint, ToPoint });
        } else {
          revng_assert(!Corners.contains({ From, To }));

          auto FromLane = From->Center.X;
          if (auto It = Lanes.Exits.find(From); It != Lanes.Exits.end()) {
            EdgeDestinationView View(To, *Label);
            revng_assert(It->second.contains(View));

            auto ExitIndex = float(It->second.at(View));
            auto CenteredIndex = ExitIndex - float(It->second.size() - 1) / 2;

            auto FromLaneGap = EdgeDistance / 2;
            if (It->second.size() != 0) {
              auto AlternativeGap = From->Size.W / 2 / It->second.size();
              if (AlternativeGap < FromLaneGap)
                FromLaneGap = AlternativeGap;
            }

            FromLane += FromLaneGap * CenteredIndex;
          }

          auto FromPoint = Point{ FromLane, FromBottom };
          auto CenterPoint = Point{ FromLane, To->Center.Y };
          auto ToPoint = Point{ To->Center.X, ToBottom };

          Corners.emplace(NodePair{ From, To },
                          Corner{ FromPoint, CenterPoint, ToPoint });
        }
      }
    }
  }

  return Corners;
}

/// A helper class used for construction of `RoutableEdge`s.
class RoutableEdgeMaker {
public:
  RoutableEdgeMaker(const RankContainer &Ranks,
                    const LaneContainer &Lanes,
                    CornerContainer &&Prerouted) :
    Ranks(Ranks), Lanes(Lanes), Prerouted(std::move(Prerouted)) {}

  RoutableEdge make(NodeView From, NodeView To, InternalEdge &Label) {
    revng_assert(Label.IsBackwards == false);

    Rank ExitIndex = 0;
    Rank ExitCount = 1;
    if (auto It = Lanes.Exits.find(From); It != Lanes.Exits.end()) {
      revng_assert(It->second.contains({ To, Label }));
      ExitIndex = It->second.at({ To, Label });
      ExitCount = It->second.size();
    }

    Rank EntryIndex = 0;
    Rank EntryCount = 1;
    if (auto It = Lanes.Entries.find(To); It != Lanes.Entries.end()) {
      revng_assert(It->second.contains({ From, Label }));
      EntryIndex = It->second.at({ From, Label });
      EntryCount = It->second.size();
    }

    Rank LaneIndex = 0;
    if (auto LayerIndex = std::min(Ranks.at(From), Ranks.at(To));
        LayerIndex < Lanes.Horizontal.size()
        && Lanes.Horizontal[LayerIndex].size()) {
      EdgeView View(From, To, Label);
      if (auto Iterator = Lanes.Horizontal[LayerIndex].find(View);
          Iterator != Lanes.Horizontal[LayerIndex].end())
        LaneIndex = Iterator->second;
    }

    decltype(RoutableEdge::Prerouted) CurrentRoute = std::nullopt;
    if (auto Iterator = Prerouted.find({ From, To });
        Iterator != Prerouted.end())
      CurrentRoute = std::move(Iterator->second);

    return RoutableEdge{
      .Label = &Label,
      .FromCenter = From->Center,
      .ToCenter = To->Center,
      .FromSize = From->Size,
      .ToSize = To->Size,
      .LaneIndex = LaneIndex,
      .ExitCount = ExitCount,
      .EntryCount = EntryCount,
      .CenteredExitIndex = float(ExitIndex) - float(ExitCount - 1) / 2,
      .CenteredEntryIndex = float(EntryIndex) - float(EntryCount - 1) / 2,
      .Prerouted = CurrentRoute
    };
  }

private:
  const RankContainer &Ranks;
  const LaneContainer &Lanes;
  CornerContainer &&Prerouted;
};

void restoreEdgeDirections(InternalGraph &Graph) {
  for (auto *From : Graph.nodes()) {
    for (auto Iterator = From->successor_edges().begin();
         Iterator != From->successor_edges().end();) {
      if (auto [To, Label] = *Iterator; Label->IsBackwards) {
        Label->IsBackwards = !Label->IsBackwards;
        To->addSuccessor(From, std::move(*Label));
        Iterator = From->removeSuccessor(Iterator);
      } else {
        ++Iterator;
      }
    }
  }
}

OrderedEdgeContainer orderEdges(InternalGraph &Graph,
                                CornerContainer &&Prerouted,
                                const RankContainer &Ranks,
                                const LaneContainer &Lanes) {
  OrderedEdgeContainer Result;
  RoutableEdgeMaker Maker(Ranks, Lanes, std::move(Prerouted));
  for (auto *From : Graph.nodes()) {
    if (!From->IsVirtual) {
      for (auto [To, Label] : From->successor_edges()) {
        Result.emplace_back(Maker.make(From, To, *Label));
        if (To->IsVirtual) {
          for (auto *Current : llvm::depth_first(To)) {
            if (!Current->IsVirtual)
              break;

            revng_assert(Current->successorCount() == 1);
            revng_assert(Current->predecessorCount() == 1
                         || (Current->predecessorCount() == 2
                             && Graph.hasEntryNode && Graph.getEntryNode()
                             && Graph.getEntryNode()->IsVirtual));

            auto [Next, NextLabel] = *Current->successor_edges().begin();
            Result.emplace_back(Maker.make(Current, Next, *NextLabel));
          }
        }
      }
    }
  }

  return Result;
}

void route(const OrderedEdgeContainer &OrderedListOfEdges,
           float MarginSize,
           float EdgeDistance) {
  for (auto &Edge : OrderedListOfEdges) {
    revng_assert(Edge.Label->IsRouted == false);

    if (Edge.Prerouted != std::nullopt) {
      Edge.Label->appendPoint(Edge.Prerouted->Start);
      Edge.Label->appendPoint(Edge.Prerouted->Center);
      Edge.Label->appendPoint(Edge.Prerouted->End);
    } else {
      // Looking for the lowest point of the edge
      auto ToUpperEdge = Edge.ToCenter.Y + Edge.ToSize.H / 2,
           FromUpperEdge = Edge.FromCenter.Y + Edge.FromSize.H / 2;
      float Corner = std::min(FromUpperEdge, ToUpperEdge);
      Corner += MarginSize + Edge.LaneIndex * EdgeDistance;

      // The concept of lanes extends to vertical segments, that otherwise
      // would merge at the points where multiple path join or separate.
      // Those points have to represent real nodes.

      float PerExit = float(Edge.FromSize.W) / Edge.ExitCount,
            PerEntry = float(Edge.ToSize.W) / Edge.EntryCount;
      float FromTheGap = std::min(EdgeDistance, PerExit) / 2,
            ToTheGap = std::min(EdgeDistance, PerEntry) / 2;
      float FromDisplacement = FromTheGap * Edge.CenteredExitIndex,
            ToDisplacement = ToTheGap * Edge.CenteredEntryIndex;
      float ToLane = Edge.ToCenter.X + ToDisplacement,
            ToTop = Edge.ToCenter.Y + Edge.ToSize.H / 2;

      Edge.Label->appendPoint(Edge.FromCenter.X + FromDisplacement,
                              Edge.FromCenter.Y - Edge.FromSize.H / 2);
      Edge.Label->appendPoint(Edge.FromCenter.X + FromDisplacement, Corner);
      Edge.Label->appendPoint(ToLane, Corner);
      Edge.Label->appendPoint(ToLane, ToTop);
    }

    Edge.Label->IsRouted = true;
  }
}

void routeWithStraightLines(const OrderedEdgeContainer &OrderedListOfEdges) {
  for (auto &Edge : OrderedListOfEdges) {
    revng_assert(Edge.Label->IsRouted == false);

    revng_assert(Edge.Prerouted == std::nullopt,
                 "Straight line routing doesn't support prerouted corners");

    Edge.Label->appendPoint(Edge.FromCenter.X,
                            Edge.FromCenter.Y - Edge.FromSize.H / 2);
    Edge.Label->appendPoint(Edge.ToCenter.X,
                            Edge.ToCenter.Y + Edge.ToSize.H / 2);

    Edge.Label->IsRouted = true;
  }
}
