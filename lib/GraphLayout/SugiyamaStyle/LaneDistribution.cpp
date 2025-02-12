/// \file LaneDistribution.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "InternalCompute.h"

struct SortableEdge {
private:
  EdgeView Edge;
  bool IsFacingRight;

  static bool
  isFacingRight(NodeView From, NodeView To, const InternalEdge &Label) {
    if (To->Center.X == From->Center.X)
      return false;

    return Label.IsBackwards != (From->Center.X > To->Center.X);
  }

public:
  SortableEdge(NodeView From, NodeView To, const InternalEdge &Label) :
    Edge(From, To, Label), IsFacingRight(isFacingRight(From, To, Label)) {}

  const EdgeView &view() const { return Edge; }
  EdgeView &&view() { return std::move(Edge); }

  /// Compares two edges. This is used as a comparator for horizontal edge lane
  /// sorting. The direction of an edge is the most most important
  /// characteristic since we want to split "left-to-right" edges from
  /// "right-to-left" ones - that helps to minimize the number of crossings.
  /// When directions are the same, edges are sorted based on the horizontal
  /// coordinates of their ends (the edge that needs to go further is placed
  /// closer to the outside of the lane section).
  bool operator<(const SortableEdge &Another) const {
    if (IsFacingRight == Another.IsFacingRight) {
      // Both edges face in the same direction.
      auto &&[LHSMin, LHSMax] = std::minmax(Edge.From->Center.X,
                                            Edge.To->Center.X);
      auto &&[RHSMin, RHSMax] = std::minmax(Another.Edge.From->Center.X,
                                            Another.Edge.To->Center.X);

      if (IsFacingRight)
        return LHSMax == RHSMax ? LHSMin < RHSMin : LHSMax < RHSMax;
      else
        return LHSMax == RHSMax ? LHSMin > RHSMin : LHSMax > RHSMax;
    } else {
      // Both edges face in different directions.
      return IsFacingRight < Another.IsFacingRight;
    }
  }
};

struct EdgeDestination {
  NodeView Neighbor;
  InternalEdge *Label = nullptr;

  EdgeDestination(NodeView Neighbor, InternalEdge &Label) :
    Neighbor(Neighbor), Label(&Label) {}

  EdgeDestinationView view() const {
    return EdgeDestinationView(Neighbor, *Label);
  }
};

LaneContainer assignLanes(InternalGraph &Graph,
                          const SegmentContainer &LinearSegments,
                          const LayoutContainer &Layout) {
  // Stores edges that require a horizontal section grouped by the layer rank.
  std::vector<llvm::SmallVector<SortableEdge, 16>> Horizontal;

  // Stores edges entering a node grouped by the node they enter.
  std::unordered_map<NodeView, llvm::SmallVector<EdgeDestination, 4>> Entries;

  // Stores edges leaving a node grouped by the node they leave.
  std::unordered_map<NodeView, llvm::SmallVector<EdgeDestination, 4>> Exits;

  // Calculate the number of lanes needed for each layer
  for (auto *From : Graph.nodes()) {
    for (auto &&[To, Label] : From->successor_edges()) {
      // If the ends of an edge are not a part of the same linear segment or
      // their horizontal coordinates are not aligned, a bend is necessary.
      if (LinearSegments.at(From) != LinearSegments.at(To)
          || From->Center.X != To->Center.X) {
        auto LayerIndex = std::min(Layout.at(From).Layer, Layout.at(To).Layer);
        if (LayerIndex >= Horizontal.size())
          Horizontal.resize(LayerIndex + 1);

        NodeView Entry = Label->IsBackwards ? To : From;
        NodeView Exit = Label->IsBackwards ? From : To;
        Horizontal[LayerIndex].emplace_back(Entry, Exit, *Label);
        Entries[Exit].emplace_back(Entry, *Label);
        Exits[Entry].emplace_back(Exit, *Label);
      }
    }
  }

  LaneContainer Result;

  // Define a comparator used for sorting entries and exits.
  struct Comparator {
    const LayoutContainer &Layout;
    NodeView FromNode;

    bool operator()(const EdgeDestination &LHS,
                    const EdgeDestination &RHS) const {
      const auto &Left = LHS.Label->IsBackwards ? FromNode : LHS.Neighbor;
      const auto &Right = RHS.Label->IsBackwards ? FromNode : RHS.Neighbor;
      return Layout.at(Left).Index < Layout.at(Right).Index;
    }
  };

  // Sort edges when they leave nodes
  for (auto &[Node, Neighbors] : Exits) {
    std::sort(Neighbors.begin(), Neighbors.end(), Comparator{ Layout, Node });

    auto &NodeExits = Result.Exits[Node];
    for (size_t ExitRank = 0; ExitRank < Neighbors.size(); ExitRank++) {
      auto &&[_, Success] = NodeExits.try_emplace(Neighbors[ExitRank].view(),
                                                  ExitRank);
      revng_assert(Success);
    }
  }

  // Sort edges where they enter nodes
  for (auto &[Node, Neighbors] : Entries) {
    std::sort(Neighbors.begin(), Neighbors.end(), Comparator{ Layout, Node });

    auto &NodeEntries = Result.Entries[Node];
    for (size_t EntryRank = 0; EntryRank < Neighbors.size(); EntryRank++) {
      auto &&[_, Success] = NodeEntries.try_emplace(Neighbors[EntryRank].view(),
                                                    EntryRank);
      revng_assert(Success);
    }
  }

  // Sort horizontal lanes
  Result.Horizontal.resize(Horizontal.size());
  for (size_t Index = 0; Index < Horizontal.size(); ++Index) {
    auto &CurrentLane = Horizontal[Index];
    // The edges going to the left and the edges going to the right are
    // distinguished in order to minimize the number of crossings.
    //
    // "left-to-right" edges need to be layered from the closest to the most
    // distant one, while "right-to-left" edges - in the opposite order.
    std::sort(CurrentLane.begin(), CurrentLane.end());
    for (size_t I = 0; I < CurrentLane.size(); I++)
      Result.Horizontal[Index][CurrentLane[I].view()] = CurrentLane.size() - I;
  }

  return Result;
}
