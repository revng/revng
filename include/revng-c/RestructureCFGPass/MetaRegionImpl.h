/// \file MetaRegion.h

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// std includes
#include <memory>
#include <set>
#include <utility>
#include <vector>

// local libraries includes
#include "revng-c/RestructureCFGPass/BasicBlockNode.h"
#include "revng-c/RestructureCFGPass/MetaRegion.h"

template<class NodeT>
void MetaRegion<NodeT>::replaceNodes(BasicBlockNodeTUPVect &N) {
  Nodes.erase(Nodes.begin(), Nodes.end());
  for (std::unique_ptr<BasicBlockNodeT> &Node : N)
    Nodes.insert(Node.get());
}

template<class NodeT>
void MetaRegion<NodeT>::updateNodes(BasicBlockNodeTSet &Removal,
                                    BasicBlockNodeT *Collapsed,
                                    BasicBlockNodeTVect &Dispatcher,
                                    BasicBlockNodeTVect &DefaultEntrySet) {
  // Remove the old SCS nodes
  bool NeedSubstitution = false;
  for (BasicBlockNodeT *Node : Removal) {
    if (Nodes.count(Node) != 0) {
      Nodes.erase(Node);
      NeedSubstitution = true;
    }
  }

  // Add the collapsed node.
  if (NeedSubstitution) {
    Nodes.insert(Collapsed);
    Nodes.insert(Dispatcher.begin(), Dispatcher.end());
    Nodes.insert(DefaultEntrySet.begin(), DefaultEntrySet.end());
  }
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
  BasicBlockNodeTVect Intersection;
  BasicBlockNodeTSet &OtherNodes = Other.getNodes();

  std::set_intersection(Nodes.begin(),
                        Nodes.end(),
                        OtherNodes.begin(),
                        OtherNodes.end(),
                        std::back_inserter(Intersection));

  return (Intersection.size() != 0);
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
