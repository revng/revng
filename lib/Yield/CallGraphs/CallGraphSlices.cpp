/// \file Slices.cpp

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <unordered_map>

#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"

#include "revng/Yield/CallGraphs/CallGraphSlices.h"

using Graph = yield::calls::PreLayoutGraph;
using Node = yield::calls::PreLayoutNode;

static Node *copyNode(yield::calls::PreLayoutGraph &Graph, const Node *Source) {
  return Graph.addNode(std::make_unique<Node>(Source->data()));
}

/// \tparam NV local `NodeView` specialization
/// \tparam INV inverted location `NodeView` specialization
template<typename NV, typename INV>
Graph makeTreeImpl(const Graph &Input, std::string_view SlicePointLocation) {
  auto SlicePointPredicate = [&SlicePointLocation](const Node *Node) {
    return Node->getLocationString() == SlicePointLocation;
  };
  auto Entry = llvm::find_if(Input.nodes(), SlicePointPredicate);
  revng_assert(Entry != Input.nodes().end());

  // Find the rank of each node, such that for any node its rank is equal to
  // the highest rank among its children plus one.
  llvm::ReversePostOrderTraversal ReversePostOrder(NV{ *Entry });
  std::unordered_map<const Node *, size_t> Ranks;
  for (const Node *CurrentNode : ReversePostOrder) {
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
  std::unordered_map<const Node *, const Node *> RealEdges;
  for (const Node *Current : ReversePostOrder) {
    auto NodeIt = Ranks.find(Current);
    revng_assert(NodeIt != Ranks.end());

    // Select the neighbour with the highest possible rank that is still
    // lower than the current node's rank.
    const Node *SelectedNeighbour = nullptr;
    size_t SelectedNeighbourRank = 0;
    for (const Node *Neighbour : llvm::children<INV>(Current)) {
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

    auto [_, Success] = RealEdges.try_emplace(Current, SelectedNeighbour);
    revng_assert(Success);
  }

  Graph Result;
  std::unordered_map<const Node *, Node *> Lookup;

  // Returns the version of the node from the new graph if it exists,
  // or adds a new one to if it does not.
  auto FindOrAddHelper = [&Result, &Lookup](const Node *OldNode) {
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
  for (const Node *Node : llvm::breadth_first(NV{ *Entry })) {
    for (auto Neighbour : llvm::children<INV>(Node)) {
      if (Ranks.contains(Neighbour)) {
        auto *NewNeighbour = FindOrAddHelper(Neighbour);
        if (RealEdges.at(Node) == Neighbour) {
          // Emit a real edge, if this is the neighbour selected earlier.
          NewNeighbour->addSuccessor(FindOrAddHelper(Node));
        } else {
          // Emit a fake node otherwise.
          auto NewNode = copyNode(Result, Node);
          NewNode->IsShallow = true;
          NewNeighbour->addSuccessor(NewNode);
        }
      }
    }
  }

  return Result;
}

yield::calls::PreLayoutGraph
yield::calls::makeCalleeTree(const PreLayoutGraph &Input,
                             std::string_view SlicePoint) {
  // Forwards direction, makes sure no successor relation ever gets lost.
  return makeTreeImpl<const PreLayoutNode *,
                      llvm::Inverse<const PreLayoutNode *>>(Input, SlicePoint);
}

yield::calls::PreLayoutGraph
yield::calls::makeCallerTree(const PreLayoutGraph &Input,
                             std::string_view SlicePoint) {
  // Backwards direction, makes sure no predecessor relation ever gets lost.
  return makeTreeImpl<llvm::Inverse<const PreLayoutNode *>,
                      const PreLayoutNode *>(Input, SlicePoint);
}
