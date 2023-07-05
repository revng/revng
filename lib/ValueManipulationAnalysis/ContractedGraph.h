#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

#include "TypeFlowGraph.h"
#include "TypeFlowNode.h"

namespace vma {

/// Super-node that contains TypeFlowGraph nodes
struct ContractedNode {
  /// The initial content of the node
  llvm::SmallSetVector<TypeFlowNode *, 8> InitialNodes;
  /// Additional nodes that have been collapsed inside this super-node
  llvm::SmallSetVector<TypeFlowNode *, 8> AdditionalNodes;

  /// Reset the content to the initialization value
  void reset() { AdditionalNodes.clear(); }

  /// Count the number of TypeFlowNodes collapsed inside this super-node
  unsigned totalSize() { return InitialNodes.size() + AdditionalNodes.size(); }

  /// Check if a TypeFlowNode is part of this super node
  bool contains(TypeFlowNode *TFN) const {
    return InitialNodes.contains(TFN) or AdditionalNodes.contains(TFN);
  }
};

/// Graph of super-nodes built upon the TypeFlowGraph
struct ContractedGraph {
  using EdgeT = std::pair<TypeFlowNode *, TypeFlowNode *>;
  using EdgeContainerT = llvm::SmallVector<EdgeT, 8>;
  using NodeContainerT = llvm::SmallVector<std::unique_ptr<ContractedNode>, 8>;
  using ReverseMapT = llvm::SmallDenseMap<TypeFlowNode *, ContractedNode *>;

  /// Color that is being decided by contracting this graph
  const ColorSet Color;

  /// Container of the ContractedNodes, owned by the graph
  NodeContainerT Nodes;
  /// Number of unique TypeFlowNodes inside all the ContractedNodes
  size_t NTypeFlowNodes = 0;
  /// Set of TypeFlowNodes that will eventually be assigned to Color
  ContractedNode *NodesToColor;
  /// Set of TypeFlowNodes from which Color will be eventually removed
  ContractedNode *NodesToUncolor;

  /// Maps each TypeFlowNode to the ContractedNode it is currently in
  ReverseMapT ReverseMap;

  /// List of edges of the TypeFlowGraph that we want to contract
  EdgeContainerT EdgeList;
  /// Edges that have been already collapsed are moved after this index
  unsigned NActiveEdges = 0;

  ContractedGraph() = delete;
  ContractedGraph(const ColorSet &CurColor) : Color(CurColor){};
  ContractedGraph(const TypeFlowGraph &N) = delete;
  ContractedGraph(ContractedGraph &&N) = delete;
  ContractedGraph &operator=(const ContractedGraph &N) = delete;
  ContractedGraph &operator=(ContractedGraph &&N) = delete;

  /// Add a new ContractedNode with an (optional) initial content
  ContractedNode *addNode(std::optional<TypeFlowNode *> InitialContent);

  /// Returns a reference to the corresponding entry in the ReverseMap
  ContractedNode *&getMapEntry(TypeFlowNode *TFGNode) {
    const auto &MapIt = ReverseMap.find(TFGNode);

    revng_assert(MapIt != ReverseMap.end());
    return MapIt->second;
  }

  /// Contract the edge at \a EdgeIndex
  ///
  /// One of the two nodes of the edge is collapsed into the other, transferring
  /// all its content and incoming/outgoing edge to the node it is collapsed
  /// with. The rules for choosing which node to collapse and which to keep are:
  /// 1. If both nodes are special, don't collapse
  /// 2. If one node is a special node, always collapse the other
  /// 3. If none of the two is special, keep the one with more nodes inside
  /// Note that \a EdgeIndex must be less that \a NActiveEdges, meaning that the
  /// collapsed edge must be active.
  void contract(size_t EdgeIndex);
  /// Reset each Node and the ReverseMap to the original state
  void reset();

  /// Check the consistency of the Graph
  ///\note Expensive Check
  void check();
};

/// Return a ContractedGraph built starting from an \a Entry Node
///
/// This function identifies a connected component of undecided nodes
/// that all have Color as a candidate. The ContractedGraph shall contain a
/// ContractedNode for each of these nodes. The neighbors of these nodes will
/// also be added in the graph, inside one of the two special ContractedNodes
/// (NodesToColo or NodesToUncolor).
///\param Entry The ContractedGraph will we built starting from this node
///\param CurColor The color which has to be assigned when contracting
void makeContractedGraph(ContractedGraph &G,
                         TypeFlowNode *Entry,
                         const ColorSet CurColor);

} // namespace vma
