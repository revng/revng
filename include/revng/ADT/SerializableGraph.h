#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/GraphTraits.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/ADT/KeyedObjectContainer.h"
#include "revng/ADT/SortedVector.h"
#include "revng/TupleTree/Introspection.h"
#include "revng/TupleTree/TupleTree.h"

//
// SerializableEdge
//
template<typename NodeType, typename EdgeLabel>
class SerializableEdge {
private:
  using KOC = KeyedObjectTraits<NodeType>;

public:
  using NodeKey = decltype(KOC::key(std::declval<NodeType>()));

public:
  SerializableEdge(NodeKey Neighbor) : Neighbor(Neighbor) {}
  SerializableEdge(NodeKey Neighbor, const EdgeLabel &Label) :
    Neighbor(Neighbor), Label(Label) {}

public:
  bool operator==(const SerializableEdge &Other) const = default;
  bool operator<(const SerializableEdge &Other) const = default;

public:
  NodeKey Neighbor;
  EdgeLabel Label;
};

template<typename NodeType, typename EdgeLabel>
struct KeyedObjectTraits<SerializableEdge<NodeType, EdgeLabel>> {
private:
  using KOC = KeyedObjectTraits<NodeType>;

public:
  using NodeKey = decltype(KOC::key(std::declval<NodeType>()));

public:
  static NodeKey key(const SerializableEdge<NodeType, EdgeLabel> &Node) {
    return KOC::key(NodeType(Node.Neighbor));
  }

  static SerializableEdge<NodeType, EdgeLabel> fromKey(const NodeKey &Key) {
    return SerializableEdge<NodeType, EdgeLabel>({ Key, {} });
  }
};

template<typename NodeType, typename EdgeLabel>
struct llvm::yaml::MappingTraits<SerializableEdge<NodeType, EdgeLabel>>
  : public TupleLikeMappingTraits<SerializableEdge<NodeType, EdgeLabel>> {};

//
// SerializableNode
//
template<typename NodeType, typename EdgeLabel>
class SerializableNode {
private:
  using KOC = KeyedObjectTraits<NodeType>;

public:
  using NodeKey = decltype(KOC::key(std::declval<NodeType>()));

public:
  bool operator==(const SerializableNode &O) const = default;

public:
  NodeType Node;
  SortedVector<SerializableEdge<NodeType, EdgeLabel>> Successors;
};

template<typename NodeType, typename EdgeLabel>
struct KeyedObjectTraits<SerializableNode<NodeType, EdgeLabel>> {
private:
  using KOC = KeyedObjectTraits<NodeType>;

public:
  using NodeKey = decltype(KOC::key(std::declval<NodeType>()));

public:
  static NodeKey key(const SerializableNode<NodeType, EdgeLabel> &Node) {
    return KOC::key(NodeType(Node.Node));
  }

  static SerializableNode<NodeType, EdgeLabel> fromKey(const NodeKey &Key) {
    return SerializableNode<NodeType, EdgeLabel>({ Key, {} });
  }
};

template<typename NodeType, typename EdgeLabel>
struct llvm::yaml::MappingTraits<SerializableNode<NodeType, EdgeLabel>>
  : public TupleLikeMappingTraits<SerializableNode<NodeType, EdgeLabel>> {};

//
// SerializableGraph
//
template<typename NodeType, typename EdgeLabel>
class SerializableGraph {
private:
  using KOC = KeyedObjectTraits<NodeType>;

public:
  using NodeKey = decltype(KOC::key(std::declval<NodeType>()));

public:
  bool operator==(const SerializableGraph &O) const = default;

public:
  template<typename GenericGraphNodeType>
  GenericGraph<GenericGraphNodeType> toGenericGraph() const {
    GenericGraph<GenericGraphNodeType> Ret;

    std::map<NodeKey, GenericGraphNodeType *> Map;

    for (const auto &N : Nodes)
      Map[KOC::key(N.Node)] = Ret.addNode(N.Node);

    for (const auto &N : Nodes)
      for (const auto &S : N.Successors)
        Map[KOC::key(N.Node)]->addSuccessor(Map[S.Neighbor], S.Label);

    if constexpr (GenericGraph<NodeType>::hasEntryNode) {
      Ret.setEntryNode(Map[EntryNode]);
    }

    return Ret;
  }

public:
  SortedVector<SerializableNode<NodeType, EdgeLabel>> Nodes;
  NodeKey EntryNode;
};

template<typename NodeType, typename EdgeLabel>
struct llvm::yaml::MappingTraits<SerializableGraph<NodeType, EdgeLabel>>
  : public TupleLikeMappingTraits<SerializableGraph<NodeType, EdgeLabel>> {};

template<typename G>
SerializableGraph<typename G::Node::NodeData, typename G::Node::EdgeLabelData>
toSerializable(const G &Graph) {
  using Node = typename G::Node::NodeData;
  using EdgeLabelData = typename G::Node::EdgeLabelData;
  using KOC = KeyedObjectTraits<Node>;
  SerializableGraph<Node, EdgeLabelData> Result;

  {
    auto Inserter = Result.Nodes.batch_insert();
    for (const auto &N : Graph.nodes())
      Inserter.insert({ KOC::key(*N), {} });
  }

  for (const auto &N : Graph.nodes()) {
    auto Inserter = Result.Nodes.at(KOC::key(*N)).Successors.batch_insert();
    for (const auto &J : N->successor_edges())
      Inserter.insert({ KOC::key(*J.Neighbor), J });
  }

  if constexpr (GenericGraph<Node>::hasEntryNode) {
    if (Graph.getEntryNode() != nullptr)
      Result.EntryNode = KOC::key(*Graph.getEntryNode());
  }

  return Result;
}

//
// Make `struct Empty` serialiazible
//

template<>
struct std::tuple_size<Empty> : std::integral_constant<size_t, 0> {};

template<>
struct TupleLikeTraits<Empty> {
  static constexpr const llvm::StringRef Name = "Empty";
  static constexpr const llvm::StringRef FullName = "Empty";
  using tuple = std::tuple<>;
  static constexpr std::array<llvm::StringRef, 0> FieldNames{};
  enum class Fields {
  };
};

template<>
struct llvm::yaml::MappingTraits<Empty> : public TupleLikeMappingTraits<Empty> {
};

//
// Expose macros to make SerializableGraph actually serializable
//
template<typename T>
struct argument_type;
template<typename T, typename U>
struct argument_type<T(U)> {
  using type = U;
};
#define TYPE(A) argument_type<void(A)>::type

#define SERIALIZABLEGRAPH_INTROSPECTION(A, B)                        \
  INTROSPECTION(TYPE((SerializableGraph<A, B>) ), Nodes, EntryNode); \
  INTROSPECTION(TYPE((SerializableNode<A, B>) ), Node, Successors);  \
  INTROSPECTION(TYPE((SerializableEdge<A, B>) ), Neighbor, Label)
