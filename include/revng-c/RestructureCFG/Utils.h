#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdlib>
#include <fstream>
#include <memory>
#include <set>

#include "revng/MFP/MFP.h"
#include "revng/MFP/SetLattices.h"

#include "revng-c/RestructureCFG/ASTNode.h"
#include "revng-c/RestructureCFG/BasicBlockNodeBB.h"

// TODO: move the definition of this object in an unique place, to avoid using
// an extern declaration
extern Logger<> CombLogger;

template<class NodeT>
using RegionCFGEdge = typename BasicBlockNode<NodeT>::EdgeDescriptor;

template<class NodeT>
inline void
moveEdgeTarget(RegionCFGEdge<NodeT> Edge, BasicBlockNode<NodeT> *NewTarget) {
  auto &SuccEdgeWithLabels = Edge.first->getSuccessorEdge(Edge.second);
  SuccEdgeWithLabels.first = NewTarget;

  auto PredEdgeWithLabels = Edge.second->extractPredecessorEdge(Edge.first);
  NewTarget->addLabeledPredecessor(PredEdgeWithLabels);
}

template<class NodeT>
inline void
moveEdgeSource(RegionCFGEdge<NodeT> Edge, BasicBlockNode<NodeT> *NewSource) {
  auto SuccEdgeWithLabels = Edge.first->extractSuccessorEdge(Edge.second);
  NewSource->addLabeledSuccessor(SuccEdgeWithLabels);

  auto &PredEdgeWithLabels = Edge.second->getPredecessorEdge(Edge.first);
  PredEdgeWithLabels.first = NewSource;
}

template<class BBNodeT>
inline void addEdge(std::pair<BBNodeT *, BBNodeT *> New,
                    const typename BBNodeT::EdgeInfo &EdgeInfos) {

  New.first->addLabeledSuccessor(std::make_pair(New.second, EdgeInfos));
  New.second->addLabeledPredecessor(std::make_pair(New.first, EdgeInfos));
}

template<class BBNodeT>
inline void addPlainEdge(std::pair<BBNodeT *, BBNodeT *> New) {
  addEdge(New, typename BBNodeT::EdgeInfo());
}

template<class BBNodeT>
inline typename BBNodeT::node_edgeinfo_pair
extractLabeledEdge(std::pair<BBNodeT *, BBNodeT *> Edge) {
  Edge.second->removePredecessor(Edge.first);
  return Edge.first->extractSuccessorEdge(Edge.second);
}

template<class BBNodeT>
inline void markEdgeInlined(std::pair<BBNodeT *, BBNodeT *> Edge) {

  // TODO: Marking the edge as inlined by temporarily removing it from the graph
  //       and re-inserting it could cause problems in reordering true/false
  //       branches for conditional nodes. Consider adding a primitive for
  //       getting a reference to the `EdgeInfo` struct of an edge without
  //       having to remove it.

  // Take care of the predecessor edge.
  auto &EdgePairBack = Edge.second->getPredecessorEdge(Edge.first);
  EdgePairBack.second.Inlined = true;

  // Take care of the successor edge.
  auto &EdgePairForw = Edge.first->getSuccessorEdge(Edge.second);
  EdgePairForw.second.Inlined = true;

  // Ensure that the forward and backward edgeinfos carry the same information.
  // For this to work, we default the spaceship operator for having the equality
  // between EdgeInfo structs.
  revng_assert(EdgePairBack.second == EdgePairForw.second);
}

template<class BBNodeT>
inline bool isEdgeInlined(std::pair<const BBNodeT *, const BBNodeT *> Edge) {

  // Take care of the backward edge.
  auto EdgePairBack = Edge.second->getPredecessorEdge(Edge.first);

  // Take care of the forward edge.
  auto EdgePairForw = Edge.first->getSuccessorEdge(Edge.second);
  bool InlinedForw = EdgePairForw.second.Inlined;

  // Ensure that the forward and backward edgeinfos carry the same information.
  revng_assert(EdgePairBack.second == EdgePairForw.second);
  return InlinedForw;
}

template<class NodeT>
inline bool
containsSmallVector(llvm::SmallVectorImpl<BasicBlockNode<NodeT> *> &Vec,
                    BasicBlockNode<NodeT> *Node) {
  for (BasicBlockNode<NodeT> *N : Vec) {
    if (N == Node) {
      return true;
    }
  }

  return false;
}

template<class NodeT>
using Stack = std::vector<std::pair<BasicBlockNode<NodeT> *, size_t>>;

template<class NodeT>
inline bool alreadyOnStack(Stack<NodeT> &Stack, BasicBlockNode<NodeT> *Node) {
  for (auto &StackElem : Stack) {
    if (StackElem.first == Node) {
      return true;
    }
  }

  return false;
}

template<class NodeT>
using BasicBlockNodeTSet = typename BasicBlockNode<NodeT>::BBNodeSet;

template<class NodeT>
inline bool alreadyOnStackQuick(BasicBlockNodeTSet<NodeT> &StackSet,
                                BasicBlockNode<NodeT> *Node) {
  return StackSet.contains(Node);
}
