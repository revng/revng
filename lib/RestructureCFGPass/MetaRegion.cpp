/// \file MetaRegion.h

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// std includes
#include <utility>
#include <set>
#include <memory>
#include <vector>

// local libraries includes
#include "revng-c/RestructureCFGPass/BasicBlockNode.h"
#include "MetaRegion.h"

void MetaRegion::replaceNodes(std::vector<std::unique_ptr<BasicBlockNode>> &NewNodes) {
  Nodes.erase(Nodes.begin(), Nodes.end());
  for (std::unique_ptr<BasicBlockNode> &Node : NewNodes) {
    Nodes.insert(Node.get());
  }
}

void MetaRegion::updateNodes(std::set<BasicBlockNode *> &Removal,
                             BasicBlockNode *Collapsed,
                             std::vector<BasicBlockNode *> Dispatcher) {

  // Remove the old SCS nodes
  bool NeedSubstitution = false;
  for (BasicBlockNode *Node : Removal) {
    if (Nodes.count(Node) != 0) {
      Nodes.erase(Node);
      NeedSubstitution = true;
    }
  }

  // Add the collapsed node.
  if (NeedSubstitution) {
        Nodes.insert(Collapsed);
        Nodes.insert(Dispatcher.begin(), Dispatcher.end());
  }
}

std::set<BasicBlockNode *> MetaRegion::getSuccessors() {
  std::set<BasicBlockNode *> Successors;

  for (BasicBlockNode *Node : nodes()) {
    for (BasicBlockNode *Successor : Node->successors()) {
      if (not containsNode(Successor)) {
        Successors.insert(Successor);
      }
    }
  }

  return Successors;
}

std::set<MetaRegion::EdgeDescriptor> MetaRegion::getOutEdges() {
  std::set<EdgeDescriptor> OutEdges;

  for (BasicBlockNode *Node : nodes()) {
    for (BasicBlockNode *Successor : Node->successors()) {
      if (not containsNode(Successor)) {
        OutEdges.insert(EdgeDescriptor(Node, Successor));
      }
    }
  }

  return OutEdges;
}

std::set<MetaRegion::EdgeDescriptor> MetaRegion::getInEdges() {
  std::set<EdgeDescriptor> InEdges;

  for (BasicBlockNode *Node : nodes()) {
    for (BasicBlockNode *Predecessor : Node->predecessors()) {
      if (not containsNode(Predecessor)) {
        InEdges.insert(EdgeDescriptor(Predecessor, Node));
      }
    }
  }

  return InEdges;
}

bool MetaRegion::intersectsWith(MetaRegion &Other) const {
  std::vector<BasicBlockNode *> Intersection;
  std::set<BasicBlockNode *> &OtherNodes = Other.getNodes();

  std::set_intersection(Nodes.begin(),
                        Nodes.end(),
                        OtherNodes.begin(),
                        OtherNodes.end(),
                        std::back_inserter(Intersection));

  return (Intersection.size() != 0);
}

bool MetaRegion::isSubSet(MetaRegion &Other) const {
  std::set<BasicBlockNode *> &OtherNodes = Other.getNodes();
  return std::includes(OtherNodes.begin(),
                       OtherNodes.end(),
                       Nodes.begin(),
                       Nodes.end());
}

bool MetaRegion::isSuperSet(MetaRegion &Other) const {
  std::set<BasicBlockNode *> &OtherNodes = Other.getNodes();
  return std::includes(Nodes.begin(),
                       Nodes.end(),
                       OtherNodes.begin(),
                       OtherNodes.end());
}

bool MetaRegion::nodesEquality(MetaRegion &Other) const {
  std::set<BasicBlockNode *> &OtherNodes = Other.getNodes();
  return Nodes == OtherNodes;
}
