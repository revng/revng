/// \file HorizontalPositions.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>

#include "Layout.h"

/// To determine the coordinates, we need to solve this equation system:
/// X_n : X_n > X_p(n) + W_n / 2 + W_p(n) / 2 + margin X_n : X_n = X_s(n)
///
/// It can be done iteratively until convergence, see the paper for the proof.
///
/// \note: this function is too long. It should probably be split up.
void setHorizontalCoordinates(const LayerContainer &Layers,
                              const std::vector<NodeView> &Order,
                              const SegmentContainer &LinearSegments,
                              const LayoutContainer &Layout,
                              float MarginSize) {
  // `IterationCount` can be tuned depending on the types of nodes and edges
  // \note: BFS layouts have wider layers, so it might make sense to make
  // `IterationCount` larger for them.
  size_t IterationCount = 10 + 5 * std::log2(Layers.size());

  // Clear the existing coordinates so they don't interfere
  for (auto Node : Order)
    Node->center() = { 0, 0 };

  // First, the minimal horizontal width for each layer is computed.
  //
  // Starting each layer at a proportional negative offset helps
  // with centering the layout.
  std::vector<double> MinimalLayerWidths;
  MinimalLayerWidths.resize(Layers.size());
  for (size_t Index = 0; Index < Layers.size(); ++Index)
    for (auto Node : Layers[Index])
      MinimalLayerWidths[Index] += Node->size().W + MarginSize;

  for (size_t Iteration = 0; Iteration < IterationCount; ++Iteration) {
    for (auto [Child, Parent] : LinearSegments)
      if (Child->center().X > Parent->center().X)
        Parent->center().X = Child->center().X;
      else
        Child->center().X = Parent->center().X;

    // \note: I feel like this loop can be A LOT simpler.
    std::map<Rank, double> MinimumX;
    for (auto Node : Order) {
      auto LayerIndex = Layout.at(Node).Layer;
      auto Iterator = MinimumX.lower_bound(LayerIndex);
      if (Iterator == MinimumX.end() || Iterator->first > LayerIndex) {
        auto Offset = -MinimalLayerWidths[LayerIndex] / 2;
        Iterator = MinimumX.insert(Iterator, { LayerIndex, Offset });
      }

      auto LeftX = Iterator->second + Node->size().W / 2 + MarginSize;
      if (LeftX > Node->center().X)
        Node->center().X = LeftX;
      Iterator->second = Node->center().X + Node->size().W / 2 + MarginSize;
    }
  }

  // At this point, the layout is very left heavy.
  // To conteract this, nodes are pushed to the right if their neighbours are
  // very far away and the successors are also on the right.
  // The positions are weighted based on the successor positions.
  for (size_t Iteration = 0; Iteration < IterationCount; ++Iteration) {
    // First of all, a safe right margin is computed for each segment.
    // A given coordinate is safe to use if no node in the segment would
    // intersect a neighbour

    std::map<NodeView, double> RightSegments;
    {
      // Check for segments that are on the rightmost edge of the
      // graph, these are the segments that can be moved freely.
      std::map<NodeView, bool> FreeSegments;

      for (auto [Child, Parent] : LinearSegments) {
        auto const &ChildPosition = Layout.at(Child);
        auto ChildLayerSize = Layers.at(ChildPosition.Layer).size();
        if (ChildLayerSize == ChildPosition.Index + 1) {
          auto Iterator = FreeSegments.lower_bound(Parent);
          if (Iterator == FreeSegments.end()
              || Iterator->first->Index > Parent->Index)
            FreeSegments.insert(Iterator, { Parent, true });
        } else {
          FreeSegments[Parent] = false;
        }
      }

      std::vector<double> Barycenters(Layers.size());
      for (size_t Index = 0; Index < Layers.size(); ++Index) {
        if (Layers[Index].size() != 0) {
          for (auto Node : Layers[Index])
            Barycenters[Index] += Node->center().X;
          Barycenters[Index] /= Layers[Index].size();
        }
      }

      for (size_t Index = 0; Index < Layers.size(); ++Index) {
        if (Layers[Index].size() != 0) {
          for (size_t J = 0; J < Layers[Index].size() - 1; ++J) {
            auto Current = Layers[Index][J];
            auto Next = Layers[Index][J + 1];
            double RightMargin = Next->center().X - Next->size().W / 2
                                 - MarginSize * 2 - Current->size().W / 2;
            auto Parent = LinearSegments.at(Current);
            auto Iterator = RightSegments.lower_bound(Parent);
            if (Iterator == RightSegments.end()
                || Iterator->first->Index > Parent->Index)
              RightSegments.insert(Iterator, { Parent, RightMargin });
            else if (RightMargin < Iterator->second)
              Iterator->second = RightMargin;
          }

          // Set the bounds for the rightmost node.
          //
          // Any coordinate is safe for the rightmost node, but the barycenters
          // of both layers should be taken into the account, so that
          // the layout does not become skewed
          float RightMargin = Layers[Index].back()->center().X;
          if (Index + 1 < Layers.size())
            if (auto Dlt = Barycenters[Index + 1] - Barycenters[Index]; Dlt > 0)
              RightMargin += Dlt;

          auto Parent = LinearSegments.at(Layers[Index].back());
          auto Iterator = RightSegments.lower_bound(Parent);
          if (Iterator == RightSegments.end()
              || Iterator->first->Index > Parent->Index)
            RightSegments.insert(Iterator, { Parent, RightMargin });
          else if (FreeSegments.at(Parent)) {
            if (RightMargin > Iterator->second)
              Iterator->second = RightMargin;
          } else if (RightMargin < Iterator->second) {
            Iterator->second = RightMargin;
          }
        }
      }
    }

    // Balance the graph to the right while keeping the safe right margin
    for (size_t Index = Layers.size() - 1; Index != size_t(-1); --Index) {
      if (Layers[Index].empty())
        continue;

      // First move the rightmost node.
      if (auto Node = Layers[Index].back(); !Node->isVirtual()) {
        if (Node->successorCount() > 0) {
          double Barycenter = 0, TotalWeight = 0;
          for (auto *Next : Node->successors()) {
            float Weight = Next->isVirtual() ? 1 : 10;
            // This weight as an arbitrary number used to help the algorithm
            // prioritize putting a node closer to its non-virtual successors.

            Barycenter += Weight * Next->center().X;
            TotalWeight += Weight;
          }
          Barycenter /= TotalWeight;

          if (Barycenter > Node->center().X)
            Node->center().X = Barycenter;
        }
      }

      // Then move the rest.
      for (size_t J = Layers[Index].size() - 1; J != size_t(-1); --J) {
        auto Node = Layers[Index][J];

        float LayerMargin = RightSegments.at(LinearSegments.at(Node));
        if (Node->successorCount() > 0) {
          double Barycenter = 0;
          double TotalWeight = 0;
          for (auto *Next : Node->successors()) {
            float Weight = Next->isVirtual() ? 1 : 10;
            // This weight as an arbitrary number used to help the algorithm
            // prioritize putting a node closer to its non-virtual successors.

            Barycenter += Weight * Next->center().X;
            TotalWeight += Weight;
          }
          Barycenter /= TotalWeight;

          if (Barycenter < LayerMargin)
            LayerMargin = Barycenter;
        }

        if (LayerMargin > Node->center().X)
          Node->center().X = LayerMargin;
      }
    }
  }
}
