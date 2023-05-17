#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>
#include <map>
#include <unordered_map>
#include <vector>

#include "revng/GraphLayout/SugiyamaStyle/Compute.h"

using RankingStrategy = yield::layout::sugiyama::RankingStrategy;
using Configuration = yield::layout::sugiyama::Configuration;

using Point = yield::layout::Point;
using Size = yield::layout::Size;

using Index = std::size_t;
using Rank = std::size_t;
using RankDelta = std::ptrdiff_t;

using InternalGraph = yield::layout::sugiyama::InternalGraph;
using InternalNode = InternalGraph::Node;
using InternalEdge = InternalNode::Edge;

/// A wrapper around an `InternalNode` pointer used for comparison overloading.
class NodeView {
public:
  NodeView(InternalNode *Pointer = nullptr) : Pointer(Pointer) {}

  operator InternalNode *() const {
    revng_assert(Pointer != nullptr);
    return Pointer;
  }
  InternalNode &operator*() const {
    revng_assert(Pointer != nullptr);
    return *Pointer;
  }
  InternalNode *operator->() const {
    revng_assert(Pointer != nullptr);
    return Pointer;
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  auto operator<=>(const NodeView &Another) const {
    revng_assert(Pointer != nullptr);
    revng_assert(Another.Pointer != nullptr);
    return Pointer->index() <=> Another.Pointer->index();
  }

private:
  InternalNode *Pointer;
};

namespace std {
template<>
struct hash<::NodeView> {
  auto operator()(const ::NodeView &View) const {
    return std::hash<InternalNode *>()(&*View);
  }
};
} // namespace std

/// A view onto an edge as a extension of a known node.
/// This is useful as a `second` value for a map where the `first` one
/// is the `From` node.
struct EdgeDestinationView {
  NodeView To;
  std::size_t EdgeIndex;

  EdgeDestinationView(NodeView To, const InternalEdge &Label) :
    To(To), EdgeIndex(Label.index()) {}

  // NOLINTNEXTLINE(readability-identifier-naming)
  auto operator<=>(const EdgeDestinationView &) const = default;
};

/// A generic view onto an edge. It's the same as the `EdgeDestinationView`
/// but stores the views onto both of the nodes.
struct EdgeView : EdgeDestinationView {
public:
  NodeView From;

  EdgeView(NodeView From, NodeView To, const InternalEdge &Label) :
    EdgeDestinationView(To, Label), From(From) {}

  // NOLINTNEXTLINE(readability-identifier-naming)
  auto operator<=>(const EdgeView &) const = default;

  InternalEdge &label() {
    auto Lambda = [this](auto Edge) {
      return Edge.Neighbor == To && Edge.Label->index() == EdgeIndex;
    };
    revng_assert(llvm::count_if(From->successor_edges(), Lambda) == 1);

    auto Iterator = llvm::find_if(From->successor_edges(), Lambda);
    revng_assert(Iterator != From->successor_edges().end());
    revng_assert(Iterator->Label != nullptr);
    return *Iterator->Label;
  }

  const InternalEdge &label() const {
    auto Lambda = [this](auto Edge) {
      return Edge.Neighbor == To && Edge.Label->index() == EdgeIndex;
    };
    revng_assert(llvm::count_if(From->successor_edges(), Lambda) == 1);

    auto Iterator = llvm::find_if(From->successor_edges(), Lambda);
    revng_assert(Iterator != From->successor_edges().end());
    revng_assert(Iterator->Label != nullptr);
    return *Iterator->Label;
  }
};

/// An internal data structure used to pass node ranks around.
using RankContainer = std::unordered_map<NodeView, Rank>;

/// An internal data structure used to pass around information about
/// the layers specific nodes belong to.
using LayerContainer = std::vector<std::vector<NodeView>>;

/// An internal data structure used to pass around information about the way
/// graph is split onto segments.
using SegmentContainer = std::unordered_map<NodeView, NodeView>;

/// It's used to describe the position of a single node within the layered grid.
struct LogicalPosition {
  Rank Layer; /// Rank of the layer the node is in.
  Rank Index; /// An index within the layer
};

/// It's used to describe the complete layout by storing the exact
/// position of each node relative to all the others.
using LayoutContainer = std::unordered_map<NodeView, LogicalPosition>;

/// An internal data structure used to pass around information about the lanes
/// used to route edges.
struct LaneContainer {
  /// Stores edges that require a horizontal section grouped by layer.
  std::vector<std::map<EdgeView, Rank>> Horizontal;

  /// Stores edges entering a node groped by the node they enter.
  std::unordered_map<NodeView, std::map<EdgeDestinationView, Rank>> Entries;

  /// Stores edges leaving a node grouped by the node they leave.
  std::unordered_map<NodeView, std::map<EdgeDestinationView, Rank>> Exits;
};

/// An internal data structure used to represent a corner. It stores three
/// points: center of the corner and two ends.
struct Corner {
  Point Start, Center, End;
};

/// An internal data structure used to represent a pair of nodes.
struct NodePair {
  NodeView From, To;

  // NOLINTNEXTLINE(readability-identifier-naming)
  auto operator<=>(const NodePair &) const = default;
};

/// An internal data structure used to pass around information about the routed
/// corners.
using CornerContainer = std::map<NodePair, Corner>;

/// An internal data structure used to represent all the necessary information
/// for an edge to be routed. This data is usable even after the internal graph
/// was destroyed.
struct RoutableEdge {
  InternalEdge *Label;
  Point FromCenter, ToCenter;
  Size FromSize, ToSize;
  Rank LaneIndex, ExitCount, EntryCount;
  float CenteredExitIndex, CenteredEntryIndex;
  std::optional<Corner> Prerouted;
};

/// An internal data structure used to pass around an ordered list of routed
/// edges.
using OrderedEdgeContainer = std::vector<RoutableEdge>;
