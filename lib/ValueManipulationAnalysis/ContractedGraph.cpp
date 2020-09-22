//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <memory>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/ValueManipulationAnalysis/TypeColors.h"

#include "ContractedGraph.h"

#include "TypeFlowGraph.h"
#include "TypeFlowNode.h"

using namespace vma;

ContractedNode *
ContractedGraph::addNode(std::optional<TypeFlowNode *> InitialContent = {}) {
  Nodes.push_back(std::make_unique<ContractedNode>());
  const auto &NewNode = Nodes.back();

  if (InitialContent) {
    NewNode->InitialNodes.insert(*InitialContent);
    ReverseMap[*InitialContent] = NewNode.get();
  }

  return NewNode.get();
}

void ContractedGraph::contract(size_t EdgeIndex) {
  revng_assert(EdgeIndex < NActiveEdges);

  EdgeT &E = EdgeList[EdgeIndex];
  ContractedNode *CN1 = getMapEntry(E.first);
  ContractedNode *CN2 = getMapEntry(E.second);

  // Verify that the nodes belongs to the graph (expensive)
  if (VerifyLog.isEnabled()) {
    const auto IsCN1 = [CN1](std::unique_ptr<ContractedNode> &N) {
      return N.get() == CN1;
    };
    revng_assert(llvm::find_if(Nodes, IsCN1));

    const auto IsCN2 = [CN2](std::unique_ptr<ContractedNode> &N) {
      return N.get() == CN2;
    };
    revng_assert(llvm::find_if(Nodes, IsCN2));
  }

  auto IsSpecialNode = [this](ContractedNode *CN) {
    return (CN == this->NodesToColor or CN == this->NodesToUncolor);
  };

  bool IsCN1Special = IsSpecialNode(CN1);
  bool IsCN2Special = IsSpecialNode(CN2);
  bool BothAreSpecial = IsCN1Special and IsCN2Special;
  bool NoneIsSpecial = not IsCN1Special and not IsCN2Special;

  // Never merge the two special nodes together
  if (BothAreSpecial or (CN1 == CN2)) {
    // Move the contracted edge after the active part of the list
    NActiveEdges--;
    std::swap(E, EdgeList[NActiveEdges]);
    return;
  }

  ContractedNode *NodeToKeep, *NodeToRemove;

  // Never remove a special node. If both are not special, remove the one with
  // less nodes.
  if (IsCN1Special or (NoneIsSpecial and CN1->totalSize() > CN2->totalSize())) {
    NodeToKeep = CN1;
    NodeToRemove = CN2;
  } else {
    NodeToKeep = CN2;
    NodeToRemove = CN1;
  }

  // Move content
  NodeToKeep->AdditionalNodes.set_union(NodeToRemove->InitialNodes);
  NodeToKeep->AdditionalNodes.set_union(NodeToRemove->AdditionalNodes);

  // Update Map
  for (TypeFlowNode *TFGNode : NodeToRemove->InitialNodes)
    getMapEntry(TFGNode) = NodeToKeep;
  for (TypeFlowNode *TFGNode : NodeToRemove->AdditionalNodes)
    getMapEntry(TFGNode) = NodeToKeep;

  // Move the contracted edge after the active part of the list
  NActiveEdges--;
  revng_assert(NActiveEdges < EdgeList.size());
  std::swap(E, EdgeList[NActiveEdges]);
}

void ContractedGraph::reset() {
  for (auto &CN : Nodes) {
    CN->reset();
    for (auto *TFGNode : CN->InitialNodes)
      ReverseMap[TFGNode] = CN.get();
  }

  NActiveEdges = EdgeList.size();
}

void ContractedGraph::check() {
  // Check edges
  for (auto &E : EdgeList) {
    revng_assert(E.first != nullptr);
    revng_assert(E.second != nullptr);

    // The edge must exist in the TypeFlowGraph
    revng_assert(llvm::is_contained(E.first->successors(), E.second));
    // Only one edge between these two nodes must exist
    revng_assert(1 == llvm::count(EdgeList, E));
    // Both nodes of the edge must be in the map
    revng_assert(ReverseMap.count(E.first));
    revng_assert(ReverseMap.count(E.second));
  }

  const auto IsCentralNode = [Col(this->Color)](const TypeFlowNode *TFGNode) {
    return TFGNode->isUndecided() and TFGNode->Candidates.contains(Col);
  };

  // Check map
  for (const auto &[CurNode, Contracted] : ReverseMap) {
    {
      revng_assert(CurNode != nullptr);
      revng_assert(Contracted != nullptr);

      // The map entries must be nodes of the graph
      revng_assert(1 == llvm::count_if(Nodes, [C(Contracted)](const auto &N) {
                     return N.get() == C;
                   }));
      // The map entries must be coherent
      const auto &Initial = Contracted->InitialNodes;
      const auto &Additional = Contracted->AdditionalNodes;
      revng_assert(1 == (Initial.count(CurNode) + Additional.count(CurNode)));
    }

    // The neighbors of the node should also be in the map
    for (TypeFlowNode *Succ : CurNode->successors()) {
      unsigned CNEdges = llvm::count(EdgeList, EdgeT{ CurNode, Succ });
      unsigned TFEdges = llvm::count(CurNode->successors(), Succ);

      if (IsCentralNode(CurNode) or IsCentralNode(Succ)) {
        revng_assert(ReverseMap.count(Succ));
        revng_assert(CNEdges == TFEdges);
      } else if (not IsCentralNode(CurNode) and not IsCentralNode(Succ)) {
        revng_assert(CNEdges == 0);
      }
    }
    // Check that the right nodes belong to the special nodes
    if (not IsCentralNode(CurNode)) {
      if (CurNode->Candidates.contains(Color))
        revng_assert(Contracted == NodesToColor);
      else
        revng_assert(Contracted == NodesToUncolor);
    }
  }

  // Check nodes
  for (auto &CN : Nodes) {
    revng_assert(CN != nullptr);
    for (auto *TFGNode : CN->InitialNodes) {
      revng_assert(TFGNode != nullptr);

      // Initial nodes must be unique
      revng_assert(1 == llvm::count_if(Nodes, [TFGNode](auto &CN2) {
                     return CN2.get()->InitialNodes.count(TFGNode);
                   }));

      revng_assert(ReverseMap.count(TFGNode));

      // Check that the right nodes belong to the special nodes
      if (not IsCentralNode(TFGNode)) {
        if (TFGNode->Candidates.contains(Color))
          revng_assert(CN.get() == NodesToColor);
        else
          revng_assert(CN.get() == NodesToUncolor);
      }
    }

    for (auto *TFGNode : CN->AdditionalNodes) {
      revng_assert(TFGNode != nullptr);
      revng_assert(ReverseMap.count(TFGNode));
    }
  }
}

///\brief Insert a TypeFlowNode in the right ContractedNode
///
/// If the TypeFlowNode is a "border" node (i.e. it's decided or it doesn't
/// have the right color) it will be inserted in the corresponding "special"
/// node (NodesToColor or NodesToUncolor).
/// Otherwise, a new ContractedNode is created.
static void addInitialNode(ContractedGraph &G, TypeFlowNode *TFGNode) {
  ContractedNode *Node;
  if (G.ReverseMap.find(TFGNode) != G.ReverseMap.end())
    return;

  bool HasWrongColor = not TFGNode->Candidates.contains(G.Color);

  if (HasWrongColor) {
    // Wrong color => uncolor
    Node = G.NodesToUncolor;
  } else if (TFGNode->isDecided()) {
    // Right color and decided => color
    Node = G.NodesToColor;
  } else if (TFGNode->isUndecided()) {
    // Right color and undecided => create a new node
    G.Nodes.push_back(std::make_unique<ContractedNode>());
    Node = G.Nodes.back().get();
  } else {
    // Unreachable
    revng_abort();
  }

  Node->InitialNodes.insert(TFGNode);
  G.ReverseMap[TFGNode] = Node;
  ++G.NTypeFlowNodes;
}

void vma::makeContractedGraph(ContractedGraph &G,
                              TypeFlowNode *Entry,
                              const ColorSet CurColor) {
  // Check entrypoint properties
  revng_assert(Entry->isDecided() and Entry->Candidates.contains(CurColor));

  // Create special nodes
  G.NodesToColor = G.addNode(Entry);
  G.NodesToUncolor = G.addNode();
  G.NTypeFlowNodes = 1;

  const auto IsBorderNode = [CurColor](const TypeFlowNode *TFNode) {
    return TFNode->isDecided() or TFNode->isUncolored()
           or (not TFNode->Candidates.contains(CurColor));
  };

  // Visit the TypeFlowGraph depth-first. Stop when you find a decided node of
  // any color or an undecided node that doesn't have CurColor as a candidate.
  llvm::df_iterator_default_set<TypeFlowNode *> DfsExtSet;
  for (TypeFlowNode *Reachable : llvm::depth_first_ext(Entry, DfsExtSet)) {
    // Add current node
    addInitialNode(G, Reachable);

    for (TypeFlowNode *Succ : Reachable->successors()) {
      if (IsBorderNode(Reachable) and IsBorderNode(Succ)) {
        // Mark the node as "not to visit" by putting it in the visited set
        if (llvm::all_of(Succ->successors(), IsBorderNode))
          DfsExtSet.insert(Succ);
      } else {
        // Add successor
        addInitialNode(G, Succ);
        // Add edge
        G.EdgeList.push_back({ Reachable, Succ });
      }
    }
  }
}
