/// \file LaneDistribution.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "Layout.h"

/// Detects whether Edge faces left, right or neither.
/// \note: Legacy function, could possibly be merged into its only user.
static auto getFacingDirection(const DirectedEdgeView &Edge) {
  auto &LHS = Edge.To->center().X;
  auto &RHS = Edge.From->center().X;
  return (LHS == RHS ? 0 : (Edge.IsBackwards ? 1 : -1) * (LHS > RHS ? 1 : -1));
}

/// Returns `true` if `Edge` is facing left-to-right, `false` otherwise.
static bool facesRight(const DirectedEdgeView &Edge) {
  return getFacingDirection(Edge) > 0;
}

/// Compares two edges. This is used as a comparator for horizontal edge lane
/// sorting. The direction of an edge is the most most important characteristic
/// since we want to split "left-to-right" edges from "right-to-left" ones -
/// that helps to minimize the number of crossings.
/// When directions are the same, edges are sorted based on the horizontal
/// coordinates of their ends (the edge that needs to go further is placed
/// closer to the outside of the lane section).
static bool compareHorizontalLanes(const DirectedEdgeView &LHS,
                                   const DirectedEdgeView &RHS) {
  bool LHSFacesRight = facesRight(LHS);
  bool RHSFacesRight = facesRight(RHS);
  if (LHSFacesRight == RHSFacesRight) {
    auto LHSFromX = LHS.From->center().X;
    auto RHSFromX = RHS.From->center().X;
    auto LHSToX = LHS.To->center().X;
    auto RHSToX = RHS.To->center().X;
    auto [LHSMin, LHSMax] = std::minmax(LHSFromX, LHSToX);
    auto [RHSMin, RHSMax] = std::minmax(RHSFromX, RHSToX);

    if (LHSFacesRight)
      return LHSMax == RHSMax ? LHSMin < RHSMin : LHSMax < RHSMax;
    else
      return LHSMax == RHSMax ? LHSMin > RHSMin : LHSMax > RHSMax;
  } else {
    return LHSFacesRight < RHSFacesRight;
  }
}

/// Returns the entry node of an edge taking into account whether the edge was
/// reversed or not.
static NodeView getEntry(const DirectedEdgeView &Edge) {
  return Edge.IsBackwards ? Edge.To : Edge.From;
}

/// Returns the exit node of an edge taking into account whether the edge was
/// reversed or not.
static NodeView getExit(const DirectedEdgeView &Edge) {
  return Edge.IsBackwards ? Edge.From : Edge.To;
}

LaneContainer assignLanes(InternalGraph &Graph,
                          const SegmentContainer &LinearSegments,
                          const LayoutContainer &Layout) {
  // Stores edges that require a horizontal section grouped by the layer rank.
  std::vector<llvm::SmallVector<DirectedEdgeView, 2>> Horizontal;

  // Stores edges entering a node grouped by the node they enter.
  std::unordered_map<NodeView, llvm::SmallVector<DirectedEdgeView, 2>> Entries;

  // Stores edges leaving a node grouped by the node they leave.
  std::unordered_map<NodeView, llvm::SmallVector<DirectedEdgeView, 2>> Exits;

  // Calculate the number of lanes needed for each layer
  for (auto *From : Graph.nodes()) {
    for (auto [To, Label] : From->successor_edges()) {
      // If the ends of an edge are not a part of the same linear segment or
      // their horizontal coordinates are not aligned, a bend is necessary.
      if (LinearSegments.at(From) != LinearSegments.at(To)
          || From->center().X != To->center().X) {
        auto LayerIndex = std::min(Layout.at(From).Layer, Layout.at(To).Layer);
        if (LayerIndex >= Horizontal.size())
          Horizontal.resize(LayerIndex + 1);

        if (Label->IsBackwards) {
          Horizontal[LayerIndex].emplace_back(To, From, Label->Pointer, true);
          Entries[From].emplace_back(To, From, Label->Pointer, true);
          Exits[To].emplace_back(To, From, Label->Pointer, true);
        } else {
          Horizontal[LayerIndex].emplace_back(From, To, Label->Pointer, false);
          Entries[To].emplace_back(From, To, Label->Pointer, false);
          Exits[From].emplace_back(From, To, Label->Pointer, false);
        }
      }
    }
  }

  LaneContainer Result;

  // Sort edges when they leave nodes
  auto ExitComparator = [&Layout](const DirectedEdgeView &LHS,
                                  const DirectedEdgeView &RHS) {
    return Layout.at(getExit(LHS)).Index < Layout.at(getExit(RHS)).Index;
  };
  for (auto &[Node, Edges] : Exits) {
    std::sort(Edges.begin(), Edges.end(), ExitComparator);
    auto &NodeExits = Result.Exits[Node];
    for (size_t ExitRank = 0; ExitRank < Edges.size(); ExitRank++)
      NodeExits.try_emplace(Edges[ExitRank], ExitRank);
  }

  // Sort edges where they enter nodes
  auto EntryComparator = [&Layout](const DirectedEdgeView &LHS,
                                   const DirectedEdgeView &RHS) {
    return Layout.at(getEntry(LHS)).Index < Layout.at(getEntry(RHS)).Index;
  };
  for (auto &[Node, Edges] : Entries) {
    std::sort(Edges.begin(), Edges.end(), EntryComparator);
    auto &NodeEntries = Result.Entries[Node];
    for (size_t EntryRank = 0; EntryRank < Edges.size(); EntryRank++)
      NodeEntries.try_emplace(Edges[EntryRank], EntryRank);
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
    std::sort(CurrentLane.begin(), CurrentLane.end(), compareHorizontalLanes);
    for (size_t I = 0; I < CurrentLane.size(); I++)
      Result.Horizontal[Index][CurrentLane[I]] = CurrentLane.size() - I;
  }

  return Result;
}
