#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/GenericGraph.h"

template<typename NodeType>
struct SerializableNode {
  using BDir = BidirectionalNode<NodeType>;
  using KKeyType = decltype(KeyedObjectTraits<BDir>::key(*((BDir *) NULL)));
  NodeType N;
  SortedVector<KKeyType> Successors;

  bool operator!=(const SerializableNode<NodeType> &O) const {
    return N != O.N;
  }
};

template<typename NodeType>
struct KeyedObjectTraits<SerializableNode<NodeType>> {
  using BDir = BidirectionalNode<NodeType>;
  using KKeyType = decltype(KeyedObjectTraits<BDir>::key(*((BDir *) NULL)));

  static KKeyType key(const SerializableNode<NodeType> &FD) {
    return KeyedObjectTraits<BDir>::key(BDir(FD.N));
  }
  static SerializableNode<NodeType> fromKey(KKeyType Key) {
    return SerializableNode<NodeType>({ Key, {} });
  }
};

template<typename NodeType>
struct SerializableGraph {
  using BDir = BidirectionalNode<NodeType>;
  using SNode = SerializableNode<NodeType>;
  using KKeyType = decltype(KeyedObjectTraits<BDir>::key(*((BDir *) NULL)));

  SortedVector<SNode> Nodes;

  bool operator!=(const SerializableGraph<NodeType> &O) const {
    return Nodes != O.Nodes;
  }

  static SerializableGraph toSerializable(const GenericGraph<BDir> &G) {
    SerializableGraph Out;
    {
      auto Inserter = Out.Nodes.batch_insert();
      for (const auto &N : G.nodes())
        Inserter.insert(SNode{ getKey(*N), {} });
    }

    for (const auto &N : G.nodes()) {
      auto Inserter = Out.Nodes.at(getKey(*N)).Successors.batch_insert();
      for (const auto &J : N->successors())
        Inserter.insert(getKey(*J));
    }
    return Out;
  };

  GenericGraph<BDir> fromSerializable() const {
    GenericGraph<BDir> Ret;

    std::map<KKeyType, BDir *> Map;

    for (const auto &N : Nodes)
      Map[getKey(N.N)] = Ret.addNode(N.N);

    for (const auto &N : Nodes)
      for (const auto &S : N.Successors)
        Map[getKey(N.N)]->addSuccessor(Map[S]);

    return Ret;
  }

private:
  static KKeyType getKey(const NodeType &N) {
    return KeyedObjectTraits<BDir>::key(BDir(N));
  }
};
