#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_map>
#include <variant>

#include "revng/ADT/GenericGraph.h"
#include "revng/GraphLayout/Traits.h"

namespace yield::layout::sugiyama {

class InternalGraph;

namespace detail {

struct InternalNode {
  layout::Point Center;
  layout::Size Size;

  bool IsVirtual;

private:
  size_t Index;

private:
  friend InternalGraph;
  InternalNode(size_t Index, layout::Size &&Size) :
    Center({ 0, 0 }), Size(std::move(Size)), IsVirtual(false), Index(Index) {}
  InternalNode(size_t Index) :
    Center({ 0, 0 }), Size({ 0, 0 }), IsVirtual(true), Index(Index) {}

public:
  InternalNode(const InternalNode &) = delete;
  InternalNode(InternalNode &&) = default;
  InternalNode &operator=(InternalNode &&) = default;

  size_t index() const { return Index; }
};

struct InternalEdge {
  bool IsRouted;
  bool IsBackwards;

private:
  size_t Index;

  using PathOwnership = std::unique_ptr<layout::Path>;
  using PathReference = layout::Path *;
  std::variant<PathOwnership, PathReference> Path;

private:
  // Only allow `InternalGraph` to construct these objects.
  // This helps to keep the indexing consistent.
  friend InternalGraph;

  /// Make a real edge.
  explicit InternalEdge(size_t Index) :
    IsRouted(false),
    IsBackwards(false),
    Index(Index),
    Path(std::make_unique<layout::Path>()) {}

  /// Make a virtual edge pointing to a path in another edge.
  InternalEdge(size_t Index, InternalEdge &Another, bool IsBackwards) :
    IsRouted(Another.IsRouted),
    IsBackwards(IsBackwards),
    Index(Index),
    Path(&Another.getPathImpl()) {}

  layout::Path &getPathImpl() {
    auto Visitor = []<typename Type>(Type &Value) -> PathReference {
      if constexpr (std::is_same<std::decay_t<Type>, PathOwnership>::value)
        return Value.get();
      else if constexpr (std::is_same<std::decay_t<Type>, PathReference>::value)
        return Value;
      else
        static_assert(type_always_false<Type>::value, "Unknown variant type");
    };
    return *std::visit(Visitor, Path);
  }

public:
  InternalEdge(const InternalEdge &) = delete;
  InternalEdge(InternalEdge &&) = default;
  InternalEdge &operator=(const InternalEdge &) = delete;
  InternalEdge &operator=(InternalEdge &&) = default;

  size_t index() const { return Index; }
  bool isVirtual() const {
    return std::get_if<PathOwnership>(&Path) == nullptr;
  }

  layout::Path &getPath() {
    revng_assert(!isVirtual(), "`getPath` should only be used on real edges.");
    return getPathImpl();
  }

private:
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

public:
  /// Adds a point to the edge path or modifies its last point if that's
  /// sufficient, based on their coordinates.
  void appendPoint(const layout::Point &Point) {
    appendPointImpl(getPathImpl(), Point);
  }
  void appendPoint(layout::Coordinate X, layout::Coordinate Y) {
    return appendPoint(layout::Point(X, Y));
  }
};

} // namespace detail

using InternalNode = MutableEdgeNode<detail::InternalNode,
                                     detail::InternalEdge,
                                     false>;

class InternalGraph : public GenericGraph<InternalNode, 16, true> {
public:
  using GenericGraph<InternalNode, 16, true>::GenericGraph;

private:
  size_t NodeIndexCounter = 0;
  size_t EdgeIndexCounter = 0;

public:
  template<typename NodeRef, typename EdgeRef>
  struct OutputLookups {
    std::vector<NodeRef> Nodes;
    std::vector<EdgeRef> Edges;
  };

public:
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

      auto [It, Success] = InternalLookup.try_emplace(Node,
                                                      Result.makeNode(Size));
      revng_assert(Success);
      revng_assert(It->second->Index == Lookup.Nodes.size());
      Lookup.Nodes.emplace_back(Node);
    }

    for (NodeRef From : llvm::nodes(Graph)) {
      for (EdgeRef Edge : llvm::children_edges<GraphType>(From)) {
        const auto &NewEdge = Result.makeEdge(InternalLookup.at(From),
                                              InternalLookup.at(Edge.Neighbor));
        revng_assert(NewEdge->Index == Lookup.Edges.size());
        Lookup.Edges.emplace_back(Edge);
      }
    }

    struct ResultType {
      InternalGraph Graph;
      const OutputLookups<NodeRef, EdgeRef> Lookup;
    };
    return ResultType{ .Graph = std::move(Result),
                       .Lookup = std::move(Lookup) };
  }

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
        LayoutTrait::setNodePosition(Lookup.Nodes[Node->Index],
                                     std::move(Node->Center));

    for (InternalNode *From : nodes()) {
      for (auto [To, Label] : From->successor_edges()) {
        revng_assert(Label->IsRouted == true);

        if (!Label->isVirtual())
          LayoutTrait::setEdgePath(Lookup.Edges[Label->Index],
                                   std::move(Label->getPath()));
      }
    }
  }

public:
  Node *makeNode(layout::Size Size, bool IsVirtual = false) {
    return addNode(detail::InternalNode{ NodeIndexCounter++, std::move(Size) });
  }
  Node *makeVirtualNode() {
    return addNode(detail::InternalNode{ NodeIndexCounter++ });
  }
  Node::Edge *makeEdge(Node *From, Node *To) {
    return From->addSuccessor(To, Node::Edge(EdgeIndexCounter++)).Label;
  }
  Node::Edge makeVirtualEdge(Node::Edge &Edge, bool IsBackwards) {
    return Node::Edge(EdgeIndexCounter++, Edge, IsBackwards);
  }
  Node::Edge makeVirtualEdge(Node::Edge &Edge) {
    return makeVirtualEdge(Edge, Edge.IsBackwards);
  }
};

} // namespace yield::layout::sugiyama
