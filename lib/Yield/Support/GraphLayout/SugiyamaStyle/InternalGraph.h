#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>
#include <map>
#include <unordered_map>
#include <vector>

#include "revng/Yield/Support/GraphLayout/SugiyamaStyle/Compute.h"

using RankingStrategy = yield::sugiyama::RankingStrategy;
using Configuration = yield::sugiyama::Configuration;

using ExternalGraph = yield::Graph;
using ExternalNode = ExternalGraph::Node;
using ExternalLabel = ExternalNode::Edge;

using Point = yield::Graph::Point;
using Size = yield::Graph::Size;

using Index = size_t;
using Rank = size_t;
using RankDelta = int64_t;

namespace detail {
/// A simple class for keeping a count.
class SimpleIndexCounter {
public:
  Index next() { return Count++; }

private:
  Index Count = 0;
};
} // namespace detail

/// An internal node representation. Contains a pointer to an external node,
/// an index and center/size helper members.
struct InternalNodeBase {
  using Indexer = detail::SimpleIndexCounter;

  explicit InternalNodeBase(ExternalNode *Node, Indexer &IRef) :
    Pointer(Node), Index(IRef.next()), LocalCenter{ 0, 0 }, LocalSize{ 0, 0 } {}

  // NOLINTNEXTLINE(readability-identifier-naming)
  auto operator<=>(const InternalNodeBase &Another) const {
    return Index <=> Another.Index;
  }

  Point &center() { return Pointer ? Pointer->Center : LocalCenter; }
  Point const &center() const {
    return Pointer ? Pointer->Center : LocalCenter;
  }
  Size &size() { return Pointer ? Pointer->Size : LocalSize; }
  Size const &size() const { return Pointer ? Pointer->Size : LocalSize; }

  bool isVirtual() const { return Pointer == nullptr; }

public:
  ExternalNode *Pointer;
  const Index Index;

private:
  Point LocalCenter; // Used to temporarily store positions of "fake" nodes.
  Size LocalSize; // Used to temporarily store size of "fake" nodes.
};

/// An internal edge label representation. Contains two pointers to external
/// labels: a forward-facing one and a backward-facing one.
struct InternalLabelBase {
public:
  InternalLabelBase(ExternalLabel *Label, bool IsBackwards = false) :
    Pointer(Label), IsBackwards(IsBackwards) {}

public:
  ExternalLabel *Pointer = nullptr;
  bool IsBackwards = false;
};

using InternalNode = MutableEdgeNode<InternalNodeBase, InternalLabelBase>;
using InternalLabel = InternalNode::Edge;
class InternalGraph : public GenericGraph<InternalNode, 16, true> {
  using Base = GenericGraph<InternalNode, 16, true>;

public:
  template<class... Args>
  InternalNode *addNode(Args &&...A) {
    return Base::addNode(A..., Indexer);
  }

private:
  typename InternalNode::Indexer Indexer;
};

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
    return Pointer->Index <=> Another.Pointer->Index;
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

namespace detail {

/// A generic view onto an edge. It stores views onto the `From` and `To`
/// nodes as well as free information about the edge label.
template<typename LabelType>
struct GenericEdgeView {
protected:
  static constexpr bool OwnsLabel = !std::is_pointer_v<std::decay_t<LabelType>>;
  using ParamType = std::conditional_t<OwnsLabel,
                                       std::decay_t<LabelType> &&,
                                       std::decay_t<LabelType>>;

public:
  NodeView From, To;
  LabelType Label;

  GenericEdgeView(NodeView From, NodeView To, ParamType Label) :
    From(From), To(To), Label(std::move(Label)) {}

  // NOLINTNEXTLINE(readability-identifier-naming)
  auto operator<=>(const GenericEdgeView &) const = default;

  std::remove_pointer_t<std::decay_t<LabelType>> &label() {
    if constexpr (OwnsLabel)
      return Label;
    else
      return *Label;
  }
  const std::remove_pointer_t<std::decay_t<LabelType>> &label() const {
    if constexpr (OwnsLabel)
      return Label;
    else
      return *Label;
  }
};

} // namespace detail

/// A view onto an edge. It stores `From` and `To` node views as well as a
/// label pointer.
using EdgeView = detail::GenericEdgeView<InternalLabel *>;

/// A view onto one of the edge labels. It stores `From` and `To` node views
/// as well as a pointer to an external label.
using DirectionlessEdgeView = detail::GenericEdgeView<ExternalLabel *>;

/// A view onto an edge. It stores `From` and `To` node views, a pointer to
/// an external label and a flag declaring the direction of the edge.
struct DirectedEdgeView : public DirectionlessEdgeView {
  bool IsBackwards = false;

  DirectedEdgeView(NodeView From, NodeView To, ParamType L, bool IsBackwards) :
    DirectionlessEdgeView(From, To, L), IsBackwards(IsBackwards) {}
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
  std::vector<std::map<DirectedEdgeView, Rank>> Horizontal;

  /// Stores edges entering a node groped by the node they enter.
  std::unordered_map<NodeView, std::map<DirectedEdgeView, Rank>> Entries;

  /// Stores edges leaving a node grouped by the node they leave.
  std::unordered_map<NodeView, std::map<DirectedEdgeView, Rank>> Exits;
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
  ExternalLabel *Label;
  Point FromCenter, ToCenter;
  Size FromSize, ToSize;
  Rank LaneIndex, ExitCount, EntryCount;
  float CenteredExitIndex, CenteredEntryIndex;
  std::optional<Corner> Prerouted;
};

/// An internal data structure used to pass around an ordered list of routed
/// edges.
using OrderedEdgeContainer = std::vector<RoutableEdge>;
