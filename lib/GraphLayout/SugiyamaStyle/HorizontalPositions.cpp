/// \file HorizontalPositions.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>

#include "InternalCompute.h"

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
                              float MarginSize,
                              float VirtualNodeWeight) {
  // `IterationCount` can be tuned depending on the types of nodes and edges
  // NOTE: BFS layouts have wider layers, so it might make sense to make
  // `IterationCount` larger for them.
  size_t IterationCount = 10 + 5 * std::log2(Layers.size());

  // Clear the existing coordinates so they don't interfere
  for (auto Node : Order)
    Node->Center = { 0, 0 };

  // First, the minimal horizontal width for each layer is computed.
  //
  // Starting each layer at a proportional negative offset helps
  // with centering the layout.
  std::vector<double> MinimalLayerWidths;
  MinimalLayerWidths.resize(Layers.size());
  for (size_t Index = 0; Index < Layers.size(); ++Index)
    for (auto Node : Layers[Index])
      MinimalLayerWidths[Index] += Node->Size.W + MarginSize;

  for (size_t Iteration = 0; Iteration < IterationCount; ++Iteration) {
    for (auto &&[Child, Parent] : LinearSegments)
      if (Child->Center.X > Parent->Center.X)
        Parent->Center.X = Child->Center.X;
      else
        Child->Center.X = Parent->Center.X;

    // NOTE: I feel like this loop can be A LOT simpler.
    std::map<Rank, double> MinimumX;
    for (auto Node : Order) {
      auto LayerIndex = Layout.at(Node).Layer;
      auto Iterator = MinimumX.lower_bound(LayerIndex);
      if (Iterator == MinimumX.end() || Iterator->first > LayerIndex) {
        auto Offset = -MinimalLayerWidths[LayerIndex] / 2;
        Iterator = MinimumX.insert(Iterator, { LayerIndex, Offset });
      }

      auto LeftX = Iterator->second + Node->Size.W / 2 + MarginSize;
      if (LeftX > Node->Center.X)
        Node->Center.X = LeftX;
      Iterator->second = Node->Center.X + Node->Size.W / 2 + MarginSize;
    }
  }

  // At this point, the layout is very left heavy.
  // To counteract this, nodes are pushed to the right if their neighbours are
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

      for (auto &&[Child, Parent] : LinearSegments) {
        auto const &ChildPosition = Layout.at(Child);
        auto ChildLayerSize = Layers.at(ChildPosition.Layer).size();
        if (ChildLayerSize == ChildPosition.Index + 1) {
          auto Iterator = FreeSegments.lower_bound(Parent);
          if (Iterator == FreeSegments.end()
              || Iterator->first->index() > Parent->index())
            FreeSegments.insert(Iterator, { Parent, true });
        } else {
          FreeSegments[Parent] = false;
        }
      }

      std::vector<double> Barycenters(Layers.size());
      for (size_t Index = 0; Index < Layers.size(); ++Index) {
        if (Layers[Index].size() != 0) {
          for (auto Node : Layers[Index])
            Barycenters[Index] += Node->Center.X;
          Barycenters[Index] /= Layers[Index].size();
        }
      }

      for (size_t Index = 0; Index < Layers.size(); ++Index) {
        if (Layers[Index].size() != 0) {
          for (size_t J = 0; J < Layers[Index].size() - 1; ++J) {
            auto Current = Layers[Index][J];
            auto Next = Layers[Index][J + 1];
            double RightMargin = Next->Center.X - Next->Size.W / 2
                                 - MarginSize * 2 - Current->Size.W / 2;
            auto Parent = LinearSegments.at(Current);
            auto Iterator = RightSegments.lower_bound(Parent);
            if (Iterator == RightSegments.end()
                || Iterator->first->index() > Parent->index())
              RightSegments.insert(Iterator, { Parent, RightMargin });
            else if (RightMargin < Iterator->second)
              Iterator->second = RightMargin;
          }

          // Set the bounds for the rightmost node.
          //
          // Any coordinate is safe for the rightmost node, but the barycenters
          // of both layers should be taken into the account, so that
          // the layout does not become skewed
          float RightMargin = Layers[Index].back()->Center.X;
          if (Index + 1 < Layers.size())
            if (auto Dlt = Barycenters[Index + 1] - Barycenters[Index]; Dlt > 0)
              RightMargin += Dlt;

          auto Parent = LinearSegments.at(Layers[Index].back());
          auto Iterator = RightSegments.lower_bound(Parent);
          if (Iterator == RightSegments.end()
              || Iterator->first->index() > Parent->index())
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
      if (auto Node = Layers[Index].back(); !Node->IsVirtual) {
        if (Node->successorCount() > 0) {
          double Barycenter = 0, TotalWeight = 0;
          for (auto *Next : Node->successors()) {
            float Weight = Next->IsVirtual ? 1 : VirtualNodeWeight;
            // This weight as an arbitrary number used to help the algorithm
            // prioritize putting a node closer to its non-virtual successors.

            Barycenter += Next->Center.X / Weight;
            TotalWeight += 1.f / Weight;
          }
          Barycenter /= TotalWeight;

          if (Barycenter > Node->Center.X)
            Node->Center.X = Barycenter;
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
            float Weight = Next->IsVirtual ? 1 : VirtualNodeWeight;
            // This weight as an arbitrary number used to help the algorithm
            // prioritize putting a node closer to its non-virtual successors.

            Barycenter += Next->Center.X / Weight;
            TotalWeight += 1.f / Weight;
          }
          Barycenter /= TotalWeight;

          if (Barycenter < LayerMargin)
            LayerMargin = Barycenter;
        }

        if (LayerMargin > Node->Center.X)
          Node->Center.X = LayerMargin;
      }
    }
  }
}

void setStaticOffsetHorizontalCoordinates(const LayerContainer &Layers,
                                          float MarginSize) {
  struct Subtree {
    /// Represents the required width of a subtree where `1` unit represents
    /// a width of single node.
    size_t LogicalWidth = 0;

    /// Represents the index of the node in it's layer (it's position relative
    /// to its sibling nodes).
    size_t LogicalPosition = 0;

    /// A place to store final node position, after tree widths are taken into
    /// the account.
    size_t ActualPosition = 0;
  };

  // Build a table of sub-trees for each node, where a subtree represents
  // the logical space required to fit all the successors of the node as well
  // it's logical position in relation to its siblings.
  //
  // NOTE: this only computes logical placement, so `ActualPosition` is left
  // unset.
  std::unordered_map<NodeView, Subtree> LookupTable;
  for (auto LayerInd = Layers.size() - 1; LayerInd != size_t(-1); --LayerInd) {
    for (size_t Index = 0; Index < Layers[LayerInd].size(); ++Index) {
      auto NodeView = Layers[LayerInd][Index];
      auto SubtreeIterator = LookupTable.find(NodeView);
      if (SubtreeIterator == LookupTable.end()) {
        // Only add new lookup entry if there is not one present already.

        bool Success = false;
        std::tie(SubtreeIterator, Success) = LookupTable.try_emplace(NodeView);
        revng_assert(Success);

        SubtreeIterator->second.LogicalWidth = 1;
      }

      // Fill logical positions based on the input layer structure.
      auto &CurrentSubtree = SubtreeIterator->second;
      CurrentSubtree.LogicalPosition = Index;

      // This relies on the fact that every node has at most one predecessor.
      revng_assert(NodeView->predecessorCount() <= 1);

      if (NodeView->hasPredecessors()) {
        auto *Predecessor = *NodeView->predecessors().begin();
        auto Iterator = LookupTable.find(Predecessor);
        if (Iterator == LookupTable.end()) {
          bool Success = false;
          std::tie(Iterator, Success) = LookupTable.try_emplace(Predecessor);
          revng_assert(Success);

          // If the only predecessor of this node is not in the lookup table,
          // add it while setting its width to that of the current node.
          auto &PredecessorSubtree = Iterator->second;
          PredecessorSubtree.LogicalWidth = CurrentSubtree.LogicalWidth;
        } else {
          // If there predecessor was already added (it has multiple
          // successors), just add the current width to it.
          auto &PredecessorSubtree = Iterator->second;
          PredecessorSubtree.LogicalWidth += CurrentSubtree.LogicalWidth;
        }
      }
    }
  }

  // Now that all the logical positions we decided by going over the layers
  // backwards, one more forwards facing pass is used to decide on their actual
  // placement.
  for (size_t LayerIndex = 0; LayerIndex < Layers.size(); ++LayerIndex) {
    size_t CurrentPosition = 0;
    for (size_t Index = 0; Index < Layers[LayerIndex].size(); ++Index) {
      auto &NodeView = Layers[LayerIndex][Index];
      auto SubtreeIterator = LookupTable.find(NodeView);
      revng_assert(SubtreeIterator != LookupTable.end());
      auto &CurrentSubtree = SubtreeIterator->second;

      // This relies on the fact that every node has at most one predecessor.
      revng_assert(NodeView->predecessorCount() <= 1);

      // If this node has to predecessors, consider it's `PredecessorPosition`
      // to be the first available space to the right.
      //
      // NOTE: this also sets the position of the first entry node to 0.
      size_t PredecessorPosition = CurrentPosition;
      if (NodeView->hasPredecessors()) {
        auto *Predecessor = *NodeView->predecessors().begin();
        auto Iterator = LookupTable.find(Predecessor);
        revng_assert(Iterator != LookupTable.end());

        // Otherwise, if there is a predecessor, fill the position based on
        // that.
        auto &PredecessorSubtree = Iterator->second;
        PredecessorPosition = PredecessorSubtree.ActualPosition;
      }

      // Make sure the current node is never to the left of its predecessor
      // as that space might be occupied by the predecessor's sibling's tree.
      CurrentPosition = std::max(CurrentPosition, PredecessorPosition);

      // Export the computed position into both the lookup map (for
      // the successors of this node to use) and as its final horizontal
      // coordinate.
      CurrentSubtree.ActualPosition = CurrentPosition;
      auto Width = CurrentSubtree.LogicalWidth;
      NodeView->Center.X = (Width / 2 + CurrentPosition) * MarginSize;

      // Update the current position, so that no sibling tree occupies the same
      // space.
      CurrentPosition += CurrentSubtree.LogicalWidth;
    }
  }
}
