#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/GraphTraits.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/ADT/KeyedObjectTraits.h"
#include "revng/ADT/SortedVector.h"
#include "revng/Model/TupleTree.h"

template<typename NodeType, typename EdgeLabel>
struct SerializableEdge {
  using KKeyType = decltype(
    KeyedObjectTraits<NodeType>::key(*((NodeType *) NULL)));
  SerializableEdge(KKeyType Neighbor) : Neighbor(Neighbor) {}
  SerializableEdge(KKeyType Neighbor, const EdgeLabel &Label) :
    Neighbor(Neighbor), Label(Label) {}
  KKeyType Neighbor;
  EdgeLabel Label;

  bool operator==(const SerializableEdge &Other) const = default;
  bool operator<(const SerializableEdge &Other) const = default;
};

template<typename NodeType, typename EdgeLabel>
struct llvm::yaml::MappingTraits<SerializableEdge<NodeType, EdgeLabel>>
  : public TupleLikeMappingTraits<SerializableEdge<NodeType, EdgeLabel>> {};

template<typename NodeType, typename EdgeLabel>
struct SerializableNode {
  using KKeyType = decltype(
    KeyedObjectTraits<NodeType>::key(*((NodeType *) NULL)));
  NodeType Node;
  SortedVector<SerializableEdge<NodeType, EdgeLabel>> Successors;

  bool operator==(const SerializableNode &O) const = default;
};

template<typename NodeType, typename EdgeLabel>
struct llvm::yaml::MappingTraits<SerializableNode<NodeType, EdgeLabel>>
  : public TupleLikeMappingTraits<SerializableNode<NodeType, EdgeLabel>> {};

template<typename NodeType, typename EdgeLabel>
struct KeyedObjectTraits<SerializableNode<NodeType, EdgeLabel>> {
  using KKeyType = decltype(
    KeyedObjectTraits<NodeType>::key(*((NodeType *) NULL)));

  static KKeyType key(const SerializableNode<NodeType, EdgeLabel> &Node) {
    return KeyedObjectTraits<NodeType>::key(NodeType(Node.Node));
  }

  static SerializableNode<NodeType, EdgeLabel> fromKey(const KKeyType &Key) {
    return SerializableNode<NodeType, EdgeLabel>({ Key, {} });
  }
};

template<typename NodeType, typename EdgeLabel>
struct KeyedObjectTraits<SerializableEdge<NodeType, EdgeLabel>> {
  using KKeyType = decltype(
    KeyedObjectTraits<NodeType>::key(*((NodeType *) NULL)));

  static KKeyType key(const SerializableEdge<NodeType, EdgeLabel> &Node) {
    return KeyedObjectTraits<NodeType>::key(NodeType(Node.Neighbor));
  }

  static SerializableEdge<NodeType, EdgeLabel> fromKey(const KKeyType &Key) {
    return SerializableEdge<NodeType, EdgeLabel>({ Key, {} });
  }
};

template<typename NodeType, typename EdgeLabel>
struct SerializableGraph {
  using KKeyType = decltype(
    KeyedObjectTraits<NodeType>::key(*((NodeType *) NULL)));

  SortedVector<SerializableNode<NodeType, EdgeLabel>> Nodes;
  KKeyType EntryNode;

  bool operator==(const SerializableGraph &O) const = default;

  template<typename GenericGraphNodeType>
  GenericGraph<GenericGraphNodeType> toGenericGraph() const {
    GenericGraph<GenericGraphNodeType> Ret;

    std::map<KKeyType, GenericGraphNodeType *> Map;

    for (const auto &N : Nodes)
      Map[KeyedObjectTraits<NodeType>::key(N.Node)] = Ret.addNode(N.Node);

    for (const auto &N : Nodes)
      for (const auto &S : N.Successors)
        Map[KeyedObjectTraits<NodeType>::key(N.Node)]->addSuccessor(Map[S.Neighbor], S.Label);

    if constexpr (GenericGraph<NodeType>::hasEntryNode) {
      Ret.setEntryNode(Map[EntryNode]);
    }

    return Ret;
  }
};

template<typename NodeType, typename EdgeLabel>
struct llvm::yaml::MappingTraits<SerializableGraph<NodeType, EdgeLabel>>
  : public TupleLikeMappingTraits<SerializableGraph<NodeType, EdgeLabel>> {};

template<typename G>
SerializableGraph<typename G::Node::NodeData, typename G::Node::EdgeLabelData>
toSerializable(const G &Graph) {
  using Node = typename G::Node::NodeData;
  using EdgeLabelData = typename G::Node::EdgeLabelData;
  SerializableGraph<Node, EdgeLabelData> Result;

  {
    auto Inserter = Result.Nodes.batch_insert();
    for (const auto &N : Graph.nodes())
      Inserter.insert({ KeyedObjectTraits<Node>::key(*N), {} });
  }

  for (const auto &N : Graph.nodes()) {
    auto Inserter = Result.Nodes.at(KeyedObjectTraits<Node>::key(*N))
                      .Successors.batch_insert();
    for (const auto &J : N->successor_edges())
      Inserter.insert({ KeyedObjectTraits<Node>::key(*J.Neighbor), J });
  }

  if constexpr (GenericGraph<Node>::hasEntryNode) {
    if (Graph.getEntryNode() != nullptr)
      Result.EntryNode = KeyedObjectTraits<Node>::key(*Graph.getEntryNode());
  }

  return Result;
}

template<typename T> struct argument_type;
template<typename T, typename U> struct argument_type<T(U)> { using type = U; };
#define TYPE(A) argument_type<void(A)>::type

#define SERIALIZABLEGRAPH_INTROSPECTION(A, B)                       \
  INTROSPECTION(TYPE((SerializableGraph<A, B>)), Nodes, EntryNode); \
  INTROSPECTION(TYPE((SerializableNode<A, B>)), Node, Successors);  \
  INTROSPECTION(TYPE((SerializableEdge<A, B>)), Neighbor, Label)
