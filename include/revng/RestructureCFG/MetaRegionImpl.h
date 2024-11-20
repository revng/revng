#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "llvm/ADT/STLExtras.h"

#include "revng/RestructureCFG/BasicBlockNodeBB.h"
#include "revng/RestructureCFG/MetaRegion.h"

template<class NodeT>
void MetaRegion<NodeT>::replaceNodes(BasicBlockNodeTUPVect &N) {
  Nodes.erase(Nodes.begin(), Nodes.end());
  for (std::unique_ptr<BasicBlockNodeT> &Node : N)
    Nodes.insert(Node.get());
}

template<class NodeT>
void MetaRegion<NodeT>::updateNodes(const BasicBlockNodeTSet &ToRemove,
                                    BasicBlockNodeT *Collapsed,
                                    BasicBlockNodeT *ExitDispatcher,
                                    const BasicBlockNodeTVect &DefaultEntrySet,
                                    const BasicBlockNodeTVect
                                      &DeduplicatedDummies) {
  // Remove the old SCS nodes
  for (BasicBlockNodeT *Node : ToRemove)
    Nodes.erase(Node);

  // Add the collapsed node.
  revng_assert(nullptr != Collapsed);
  Nodes.insert(Collapsed);

  // Add the exit dispatcher if present
  if (ExitDispatcher)
    Nodes.insert(ExitDispatcher);

  // Add the set nodes that come from outside if present
  revng_assert(not llvm::any_of(DefaultEntrySet, [this](BasicBlockNodeT *B) {
    return this->containsNode(B);
  }));
  Nodes.insert(DefaultEntrySet.begin(), DefaultEntrySet.end());

  // Remove deduplicated dummy nodes created during the exit dispatcher building
  for (BasicBlockNodeT *Node : DeduplicatedDummies)
    Nodes.erase(Node);
}

template<class NodeT>
std::set<BasicBlockNode<NodeT> *> MetaRegion<NodeT>::getSuccessors() {
  BasicBlockNodeTSet Successors;

  for (BasicBlockNodeT *Node : nodes()) {
    for (BasicBlockNodeT *Successor : Node->successors()) {
      if (not containsNode(Successor)) {
        Successors.insert(Successor);
      }
    }
  }

  return Successors;
}

template<class NodeT>
using EdgeDescriptor = typename MetaRegion<NodeT>::EdgeDescriptor;

template<class NodeT>
std::set<EdgeDescriptor<NodeT>> MetaRegion<NodeT>::getOutEdges() {
  std::set<EdgeDescriptor> OutEdges;

  for (BasicBlockNodeT *Node : nodes()) {
    for (BasicBlockNodeT *Successor : Node->successors()) {
      if (not containsNode(Successor)) {
        OutEdges.insert(EdgeDescriptor(Node, Successor));
      }
    }
  }

  return OutEdges;
}

template<class NodeT>
std::set<EdgeDescriptor<NodeT>> MetaRegion<NodeT>::getInEdges() {
  std::set<EdgeDescriptor> InEdges;

  for (BasicBlockNodeT *Node : nodes()) {
    for (BasicBlockNodeT *Predecessor : Node->predecessors()) {
      if (not containsNode(Predecessor)) {
        InEdges.insert(EdgeDescriptor(Predecessor, Node));
      }
    }
  }

  return InEdges;
}

template<class NodeT>
bool MetaRegion<NodeT>::intersectsWith(MetaRegion<NodeT> &Other) const {
  BasicBlockNodeTSet &OtherNodes = Other.getNodes();

  auto NodesIt = Nodes.begin();
  auto NodesEnd = Nodes.end();
  auto OtherIt = OtherNodes.begin();
  auto OtherEnd = OtherNodes.end();
  while (NodesIt != NodesEnd and OtherIt != OtherEnd) {
    if (*NodesIt < *OtherIt) {
      ++NodesIt;
    } else {
      if (not(*OtherIt < *NodesIt)) {
        return true; // This is equal, hence intersection is not empty,
                     // hence return true.
      }
      ++OtherIt;
    }
  }

  // if we reach this point no element was in common, return false
  return false;
}

template<class NodeT>
bool MetaRegion<NodeT>::isSubSet(MetaRegion<NodeT> &Other) const {
  BasicBlockNodeTSet &OtherNodes = Other.getNodes();
  return std::includes(OtherNodes.begin(),
                       OtherNodes.end(),
                       Nodes.begin(),
                       Nodes.end());
}

template<class NodeT>
bool MetaRegion<NodeT>::isSuperSet(MetaRegion<NodeT> &Other) const {
  BasicBlockNodeTSet &OtherNodes = Other.getNodes();
  return std::includes(Nodes.begin(),
                       Nodes.end(),
                       OtherNodes.begin(),
                       OtherNodes.end());
}

template<class NodeT>
bool MetaRegion<NodeT>::nodesEquality(MetaRegion<NodeT> &Other) const {
  BasicBlockNodeTSet &OtherNodes = Other.getNodes();
  return Nodes == OtherNodes;
}
