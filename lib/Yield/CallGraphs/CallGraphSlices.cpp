/// \file Slices.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_map>

#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"

#include "revng/Yield/CallGraphs/CallGraphSlices.h"

using NodeView = const yield::Graph::Node *;

static yield::Graph::Node *
copyNode(yield::Graph &Graph, const yield::Graph::Node *Source) {
  auto New = std::make_unique<yield::Graph::Node>(Source->data());
  return Graph.addNode(std::move(New));
}

/// \tparam NV local `NodeView` specialization
/// \tparam INV inverted location `NodeView` specialization
template<typename NV, typename INV>
yield::Graph
makeTreeImpl(const yield::Graph &Input, const BasicBlockID &SlicePoint) {
  auto SlicePointPredicate = [&SlicePoint](NodeView Node) {
    return Node->Address == SlicePoint;
  };
  auto Entry = llvm::find_if(Input.nodes(), SlicePointPredicate);
  revng_assert(Entry != Input.nodes().end());

  // Find the rank of each node, such that for any node its rank is equal to
  // the highest rank among its children plus one.
  llvm::ReversePostOrderTraversal ReversePostOrder(NV{ *Entry });
  std::unordered_map<const yield::Graph::Node *, size_t> Ranks;
  for (NodeView CurrentNode : ReversePostOrder) {
    uint64_t &CurrentRank = Ranks[CurrentNode];

    CurrentRank = 0;
    for (auto Child : llvm::children<INV>(CurrentNode))
      if (auto RankIterator = Ranks.find(Child); RankIterator != Ranks.end())
        CurrentRank = std::max(RankIterator->second + 1, CurrentRank);
  }

  // For each node, select a single predecessor to keep connected to.
  // The ranks calculated earlier are used to choose a specific one.
  //
  // TODO: We should consider a better selection algorithm.
  std::unordered_map<NodeView, NodeView> RealEdges;
  for (NodeView Node : ReversePostOrder) {
    auto NodeIt = Ranks.find(Node);
    revng_assert(NodeIt != Ranks.end());

    // Select the neighbour with the highest possible rank that is still
    // lower than the current node's rank.
    NodeView SelectedNeighbour = nullptr;
    size_t SelectedNeighbourRank = 0;
    for (NodeView Neighbour : llvm::children<INV>(Node)) {
      if (auto Iterator = Ranks.find(Neighbour); Iterator != Ranks.end()) {
        // If an inverse neighbour is not present in the `Ranks` table, it's not
        // a part of the desired slice, as such we can safely ignore it.
        size_t Rank = Iterator->second;
        if (Rank < NodeIt->second && Rank >= SelectedNeighbourRank) {
          SelectedNeighbour = Neighbour;
          SelectedNeighbourRank = Rank;
        }
      }
    }

    auto [_, Success] = RealEdges.try_emplace(Node, SelectedNeighbour);
    revng_assert(Success);
  }

  yield::Graph Result;
  std::unordered_map<NodeView, yield::Graph::Node *> Lookup;

  // Returns the version of the node from the new graph if it exists,
  // or adds a new one to if it does not.
  auto FindOrAddHelper = [&Result, &Lookup](NodeView OldNode) {
    if (auto NewNode = Lookup.find(OldNode); NewNode != Lookup.end())
      return NewNode->second;
    else
      return Lookup.emplace(OldNode, copyNode(Result, OldNode)).first->second;
  };

  // Manually adding `Entry` to the result graphs guarantees that it's never
  // empty. Since we only ever iterate on edges, this will guarantee that the
  // produced graph is not empty even in the cases where `Entry` has no edges.
  Result.setEntryNode(FindOrAddHelper(*Entry));

  // Fill in the `Result` graph.
  for (NodeView Node : llvm::breadth_first(NV{ *Entry })) {
    for (auto Neighbour : llvm::children<INV>(Node)) {
      if (Ranks.contains(Neighbour)) {
        auto *NewNeighbour = FindOrAddHelper(Neighbour);
        if (RealEdges.at(Node) == Neighbour) {
          // Emit a real edge, if this is the neighbour selected earlier.
          NewNeighbour->addSuccessor(FindOrAddHelper(Node));
        } else {
          // Emit a fake node otherwise.
          auto NewNode = copyNode(Result, Node);
          NewNode->NextAddress = NewNode->Address;
          NewNeighbour->addSuccessor(NewNode);
        }
      }
    }
  }

  return Result;
}

yield::Graph yield::calls::makeCalleeTree(const yield::Graph &Input,
                                          const BasicBlockID &SlicePoint) {
  // Forwards direction, makes sure no successor relation ever gets lost.
  return makeTreeImpl<NodeView, llvm::Inverse<NodeView>>(Input, SlicePoint);
}
yield::Graph yield::calls::makeCallerTree(const yield::Graph &Input,
                                          const BasicBlockID &SlicePoint) {
  // Backwards direction, makes sure no predecessor relation ever gets lost.
  return makeTreeImpl<llvm::Inverse<NodeView>, NodeView>(Input, SlicePoint);
}
