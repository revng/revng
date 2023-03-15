/// \file EdgeRouting.cpp
/// \brief

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

/// A helper class used for construction of `RoutableEdge`s.
class RoutableEdgeMaker {
public:
  RoutableEdgeMaker(const RankContainer &Ranks,
                    const LaneContainer &Lanes,
                    CornerContainer &&Prerouted) :
    Ranks(Ranks), Lanes(Lanes), Prerouted(std::move(Prerouted)) {}

  RoutableEdge make(NodeView From, NodeView To, ExternalLabel *Label) {
    auto View = DirectedEdgeView{ From, To, Label, false };

    Rank ExitIndex = 0;
    Rank ExitCount = 1;
    if (auto It = Lanes.Exits.find(From); It != Lanes.Exits.end()) {
      if (It->second.size() != 0) {
        revng_assert(It->second.contains(View));
        ExitIndex = It->second.at(View);
        ExitCount = It->second.size();
      }
    }

    Rank EntryIndex = 0;
    Rank EntryCount = 1;
    if (auto It = Lanes.Entries.find(To); It != Lanes.Entries.end()) {
      if (It->second.size() != 0) {
        revng_assert(It->second.contains(View));
        EntryIndex = It->second.at(View);
        EntryCount = It->second.size();
      }
    }

    Rank LaneIndex = 0;
    if (auto LayerIndex = std::min(Ranks.at(From), Ranks.at(To));
        LayerIndex < Lanes.Horizontal.size()
        && Lanes.Horizontal[LayerIndex].size()) {
      if (auto Iterator = Lanes.Horizontal[LayerIndex].find(View);
          Iterator != Lanes.Horizontal[LayerIndex].end())
        LaneIndex = Iterator->second;
    }

    decltype(RoutableEdge::Prerouted) CurrentRoute = std::nullopt;
    if (auto Iterator = Prerouted.find({ From, To });
        Iterator != Prerouted.end())
      CurrentRoute = std::move(Iterator->second);

    return RoutableEdge{
      .Label = Label,
      .FromCenter = From->center(),
      .ToCenter = To->center(),
      .FromSize = From->size(),
      .ToSize = To->size(),
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

OrderedEdgeContainer orderEdges(InternalGraph &&Graph,
                                CornerContainer &&Prerouted,
                                const RankContainer &Ranks,
                                const LaneContainer &Lanes) {
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

  OrderedEdgeContainer Result;
  RoutableEdgeMaker Maker(Ranks, Lanes, std::move(Prerouted));
  for (auto *From : Graph.nodes()) {
    if (!From->isVirtual()) {
      for (auto [To, Label] : From->successor_edges()) {
        revng_assert(Label->IsBackwards == false);
        Result.emplace_back(Maker.make(From, To, Label->Pointer));
        if (To->isVirtual()) {
          for (auto *Current : llvm::depth_first(To)) {
            if (!Current->isVirtual())
              break;

            revng_assert(Current->successorCount() == 1);
            revng_assert(Current->predecessorCount() == 1
                         || (Current->predecessorCount() == 2
                             && Graph.hasEntryNode && Graph.getEntryNode()
                             && Graph.getEntryNode()->isVirtual()));

            auto [Next, NextLabel] = *Current->successor_edges().begin();
            revng_assert(NextLabel->IsBackwards == false);
            Result.emplace_back(Maker.make(Current, Next, NextLabel->Pointer));
          }
        }
      }
    }
  }

  // Move the graph out of an input parameter so that it gets deleted at
  // the end of the scope of this function.
  auto GraphOnLocalStack = std::move(Graph);

  return Result;
}

/// Adds a point to the edge path or replaces its last point based
/// on their coordinates.
template<typename PathType>
void appendPoint(PathType &Path, const Point &P) {
  if (Path.size() > 1) {
    auto &First = *std::prev(std::prev(Path.end()));
    auto &Second = *std::prev(Path.end());

    auto LHS = (P.Y - Second.Y) * (Second.X - First.X);
    auto RHS = (Second.Y - First.Y) * (P.X - Second.X);
    if (LHS == RHS)
      Path.pop_back();
  }
  Path.push_back(P);
}

void route(const OrderedEdgeContainer &OrderedListOfEdges,
           float MarginSize,
           float EdgeDistance) {
  for (auto &Edge : OrderedListOfEdges) {
    if (Edge.Label->Status == ExternalGraph::EdgeStatus::Hidden)
      continue;

    if (Edge.Prerouted != std::nullopt) {
      appendPoint(Edge.Label->Path, Edge.Prerouted->Start);
      appendPoint(Edge.Label->Path, Edge.Prerouted->Center);
      appendPoint(Edge.Label->Path, Edge.Prerouted->End);
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

      appendPoint(Edge.Label->Path,
                  Point{ Edge.FromCenter.X + FromDisplacement,
                         Edge.FromCenter.Y - Edge.FromSize.H / 2 });
      appendPoint(Edge.Label->Path,
                  Point{ Edge.FromCenter.X + FromDisplacement, Corner });
      appendPoint(Edge.Label->Path, Point{ ToLane, Corner });
      appendPoint(Edge.Label->Path, Point{ ToLane, ToTop });
    }

    Edge.Label->Status = ExternalGraph::EdgeStatus::Routed;
  }
}

void routeWithStraightLines(const OrderedEdgeContainer &OrderedListOfEdges) {
  for (auto &Edge : OrderedListOfEdges) {
    if (Edge.Label->Status == ExternalGraph::EdgeStatus::Hidden)
      continue;

    revng_assert(Edge.Prerouted == std::nullopt,
                 "Straight line routing doesn't support prerouted corners");

    appendPoint(Edge.Label->Path,
                Point{ Edge.FromCenter.X,
                       Edge.FromCenter.Y - Edge.FromSize.H / 2 });
    appendPoint(Edge.Label->Path,
                Point{ Edge.ToCenter.X, Edge.ToCenter.Y + Edge.ToSize.H / 2 });

    Edge.Label->Status = ExternalGraph::EdgeStatus::Routed;
  }
}
