#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_map>
#include <variant>

#include "revng/ADT/GenericGraph.h"
#include "revng/Yield/Support/GraphLayout/Traits.h"

namespace yield::layout::sugiyama {

class InternalGraph;

namespace detail {

struct InternalNode {
  layout::Point Center;
  layout::Size Size;

  bool IsVirtual;

private:
  std::size_t Index;

public:
  InternalNode(InternalGraph &Graph, layout::Size &&Size);
  InternalNode(InternalGraph &Graph);

  ~InternalNode() = default;
  InternalNode(const InternalNode &) = delete;
  InternalNode(InternalNode &&) = default;
  InternalNode &operator=(const InternalNode &) = delete;
  InternalNode &operator=(InternalNode &&) = default;

  std::size_t index() const { return Index; }
};

/// Represents an edge in the internal graph.
struct InternalEdge {
  /// A helper field used to represent edges that were temporarily "flipped".
  ///
  /// This is used when converting an arbitrary graph into a DAG - all the edges
  /// changed by the conversion have this parameter set to `true`, which allows
  /// them to be cleanly reverted when needed.
  bool IsBackwards;

  /// A helper field used to denote edges for which the path is already
  /// finalized. Because we are unable to track the progress using the \ref Path
  /// field - it is often shared between multiple edges, so it cannot be
  /// a clear indicator for any given edge.
  ///
  /// When the layouter is done, this field must be set to `true` for every
  /// single edge in the graph. Otherwise the layout is not valid (something
  /// went horribly wrong). See \ref exportInto for the assertion of the fact.
  bool IsRouted;

private:
  std::size_t Index;

  /// Represents an owned path, it's only set to edges that have a corresponding
  /// edge in the "external" graph, this "internal" graph is constructed from.
  using PathOwner = std::unique_ptr<layout::Path>;

  /// Represents a path reference. The path in question must belong to some
  /// other edge in the same graph.
  using PathReference = layout::Path *;

  /// Because the layouter heavily relies on splitting edges into pieces
  /// (partitioning them, see `GraphPreparation.cpp` for more details), we need
  /// a way to let multiple paths accumulate into a single path that can
  /// eventually be exported for the original edge.
  ///
  /// To achieve this, each edge holds EITHER a real path (\ref PathOwner) OR
  /// just a reference to a path held by some other edge (\ref PathReference).
  std::variant<PathOwner, PathReference> Path;

public:
  /// Make a real edge.
  explicit InternalEdge(InternalGraph &Graph);

  /// Make a virtual edge pointing to a path in another edge.
  InternalEdge(InternalGraph &Graph, InternalEdge &Another, bool IsBackwards);

  ~InternalEdge() = default;
  InternalEdge(const InternalEdge &) = delete;
  InternalEdge(InternalEdge &&) = default;
  InternalEdge &operator=(const InternalEdge &) = delete;
  InternalEdge &operator=(InternalEdge &&) = default;

  std::size_t index() const { return Index; }

  /// \returns `true` for edges that own their paths, `false` otherwise.
  bool isVirtual() const { return std::holds_alternative<PathReference>(Path); }

  /// Adds a point to the edge path or modifies its last point if that's
  /// sufficient, based on their coordinates.
  void appendPoint(const layout::Point &Point) {
    appendPointImpl(getPathImpl(), Point);
  }
  void appendPoint(layout::Coordinate X, layout::Coordinate Y) {
    return appendPoint(layout::Point(X, Y));
  }

  /// Helper used to export the path. Only defined for edge owners.
  ///
  /// By explicitly forbidding this to be called on the references, we get
  /// a guarantee that each path is returned exactly once per full graph
  /// iteration.
  ///
  /// The expected way to extract the paths is
  /// ```cpp
  /// for (auto *FromNode : MyInternalGraph)
  ///   for (auto [ToNode, Edge] : FromNode->successor_edges())
  ///     if (!Edge.isVirtual())
  ///       doSomethingWithThePath(std::move(Edge.getPath()));
  /// ```
  layout::Path &getPath() {
    revng_assert(!isVirtual(), "`getPath` should only be used on real edges.");
    return getPathImpl();
  }

private:
  layout::Path &getPathImpl() {
    auto Visitor = []<typename Type>(Type &Value) -> PathReference {
      if constexpr (std::is_same<std::decay_t<Type>, PathOwner>::value)
        return Value.get();
      else if constexpr (std::is_same<std::decay_t<Type>, PathReference>::value)
        return Value;
      else
        static_assert(type_always_false<Type>::value, "Unknown variant type");
    };
    return *std::visit(Visitor, Path);
  }

  static void appendPointImpl(layout::Path &Path, const layout::Point &Point) {
    if (Path.size() > 1) {
      auto &First = *std::prev(std::prev(Path.end()));
      auto &Second = *std::prev(Path.end());

      auto LHS = (Point.Y - Second.Y) * (Second.X - First.X);
      auto RHS = (Second.Y - First.Y) * (Point.X - Second.X);
      if (LHS == RHS)
        Path.pop_back();
    }
    Path.push_back(Point);
  }
};

} // namespace detail

using InternalNode = MutableEdgeNode<detail::InternalNode,
                                     detail::InternalEdge,
                                     false>;

namespace detail {

/// The result type of the \ref make method containing the newly-built
/// graph and the lookup table necessary for the results of the layouter to be
/// merged back into the original graph.
template<typename NodeRef, typename EdgeRef>
struct InternalGraphWithLookupTable;

} // namespace detail

class InternalGraph : public GenericGraph<InternalNode, 16, true> {
public:
  using GenericGraph<InternalNode, 16, true>::GenericGraph;

private:
  std::size_t NodeIndexCounter = 0;
  std::size_t EdgeIndexCounter = 0;

public:
  std::size_t getAndIncrementNodeIndex() { return NodeIndexCounter++; }
  std::size_t getAndIncrementEdgeIndex() { return EdgeIndexCounter++; }

public:
  template<typename NodeRef, typename EdgeRef>
  struct OutputLookups {
    std::vector<NodeRef> Nodes;
    std::vector<EdgeRef> Edges;
  };

public:
  /// Split the input graph into two pieces: an internal graph the layouter
  /// requires and a lookup table allowing the results to be exported back
  /// into the original graph (see \ref exportInto).
  /// @tparam GraphType The type of the graph to split. It has to provide
  ///         `yield::layout::LayoutableGraphTraits`.
  /// @param Graph An input graph.
  /// @return A pair of the output graph and the lookup table
  ///         (see \ref InternalGraphPair)
  template<layout::HasLayoutableGraphTraits GraphType>
  static auto make(const GraphType &Graph) {
    using LLVMTrait = llvm::GraphTraits<GraphType>;
    using LayoutTrait = layout::LayoutableGraphTraits<GraphType>;

    using NodeRef = typename LLVMTrait::NodeRef;
    using EdgeRef = typename LLVMTrait::EdgeRef;

    InternalGraph Result;
    OutputLookups<NodeRef, EdgeRef> Lookup;
    Lookup.Nodes.reserve(LLVMTrait::size(Graph));

    std::unordered_map<NodeRef, InternalGraph::Node *> InternalLookup;
    for (NodeRef Node : llvm::nodes(Graph)) {
      layout::Size Size = LayoutTrait::getNodeSize(Node);

      InternalNode *NewNode = Result.addNode(Result, std::move(Size));
      auto [It, S] = InternalLookup.try_emplace(Node, NewNode);
      revng_assert(S);
      revng_assert(It->second->index() == Lookup.Nodes.size());
      Lookup.Nodes.emplace_back(Node);
    }

    for (NodeRef From : llvm::nodes(Graph)) {
      for (EdgeRef Edge : llvm::children_edges<GraphType>(From)) {
        auto *To = InternalLookup.at(Edge.Neighbor);
        auto EV = InternalLookup.at(From)->addSuccessor(To, Node::Edge(Result));
        revng_assert(EV.Label->index() == Lookup.Edges.size());
        Lookup.Edges.emplace_back(Edge);
      }
    }

    using GraphWithLookupTable = detail::InternalGraphWithLookupTable<NodeRef,
                                                                      EdgeRef>;
    return GraphWithLookupTable{ .Graph = std::move(Result),
                                 .Lookup = std::move(Lookup) };
  }

  /// Uses lookup table produced when \ref make 'ing the graph to export
  /// the results of the layouter back into it.
  ///
  /// This relies on the `LayoutableGraphTraits` to specify the way to export
  /// the data, specifically `setNodePosition` and `setEdgePath` members.
  ///
  /// @param Lookup The table produced by the \ref make method when making this
  ///        graph.
  ///
  /// @tparam NodeR The node type of the original graph. It is automatically
  ///         deduced from the \ref Lookup.
  /// @tparam EdgeR The edge type of the original graph. It is automatically
  ///         deduced from the \ref Lookup.
  /// @tparam GraphType The output graph type.
  ///
  /// \note All the virtual nodes are ignored, so for a node/edge to be
  ///       exported, it has to have been present in the original graph passed
  ///       into \ref make.
  template<layout::HasLayoutableGraphTraits GraphType,
           typename NodeR = typename llvm::GraphTraits<GraphType>::NodeRef,
           typename EdgeR = typename llvm::GraphTraits<GraphType>::EdgeRef>
  void exportInto(const OutputLookups<NodeR, EdgeR> &Lookup) {
    using LLVMTrait = llvm::GraphTraits<GraphType>;
    using LayoutTrait = layout::LayoutableGraphTraits<GraphType>;

    using NodeRef = typename LLVMTrait::NodeRef;
    using EdgeRef = typename LLVMTrait::EdgeRef;
    static_assert(std::is_same<NodeRef, NodeR>::value);
    static_assert(std::is_same<EdgeRef, EdgeR>::value);

    for (InternalNode *Node : nodes())
      if (!Node->IsVirtual)
        LayoutTrait::setNodePosition(Lookup.Nodes[Node->index()],
                                     std::move(Node->Center));

    for (InternalNode *From : nodes()) {
      for (auto [To, Label] : From->successor_edges()) {
        revng_assert(Label->IsRouted == true);

        if (!Label->isVirtual()) {
          revng_assert(!Label->getPath().empty());
          LayoutTrait::setEdgePath(Lookup.Edges[Label->index()],
                                   std::move(Label->getPath()));
        }
      }
    }
  }

public:
  /// An index agnostic way to make a new virtual node.
  ///
  /// This is the only method allowed to be used for making nodes. Please don't
  /// call `Graph.addNode` manually.
  ///
  /// \note virtual nodes are only created for the layouter's convenience, and
  ///       do not carry ANY impact on the original graph. Any information they
  ///       hold is not propagated back.
  Node *makeVirtualNode() { return addNode(*this); }

  /// An index agnostic way to make a virtual edge. It does not hold its own
  /// path object, but instead just refers to the path of the \ref Edge.
  ///
  /// It's up to the creator of this edge to ensure that the lifetime of this
  /// virtual object does not exceed the lifetime of the original edge.
  ///
  /// \note the pointer is stable, so the original edge can be moved around,
  /// just not discarded.
  Node::Edge makeVirtualEdge(Node::Edge &Edge, bool IsBackwards) {
    return Node::Edge(*this, Edge, IsBackwards);
  }
};

namespace detail {

template<typename NodeRef, typename EdgeRef>
struct InternalGraphWithLookupTable {
  InternalGraph Graph;
  const InternalGraph::OutputLookups<NodeRef, EdgeRef> Lookup;
};

inline InternalNode::InternalNode(InternalGraph &Graph, layout::Size &&Size) :
  Center({ 0, 0 }),
  Size(std::move(Size)),
  IsVirtual(false),
  Index(Graph.getAndIncrementNodeIndex()) {
}

inline InternalNode::InternalNode(InternalGraph &Graph) :
  Center({ 0, 0 }),
  Size({ 0, 0 }),
  IsVirtual(true),
  Index(Graph.getAndIncrementNodeIndex()) {
}

inline InternalEdge::InternalEdge(InternalGraph &Graph) :
  IsBackwards(false),
  IsRouted(false),
  Index(Graph.getAndIncrementEdgeIndex()),
  Path(std::make_unique<layout::Path>()) {
}

inline InternalEdge::InternalEdge(InternalGraph &Graph,
                                  InternalEdge &Another,
                                  bool IsBackwards) :
  IsBackwards(IsBackwards),
  IsRouted(Another.IsRouted),
  Index(Graph.getAndIncrementEdgeIndex()),
  Path(&Another.getPathImpl()) {
}

} // namespace detail.

} // namespace yield::layout::sugiyama
