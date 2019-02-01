/// \file BasicBlockNode.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdlib>
#include <map>
#include <set>

// Local libraries includes
#include "revng-c/RestructureCFGPass/BasicBlockNode.h"
#include "revng-c/RestructureCFGPass/RegionCFGTree.h"
#include "revng-c/RestructureCFGPass/Utils.h"

using namespace llvm;

BasicBlockNode::BasicBlockNode(RegionCFG *Parent,
                               llvm::BasicBlock * BB,
                               RegionCFG *Collapsed,
                               const std::string &Name,
                               Type T,
                               unsigned Value) :
    ID(Parent->getNewID()),
    Parent(Parent),
    BB(BB),
    CollapsedRegion(Collapsed),
    NodeType(T),
    Name(Name),
    StateVariableValue(Value) {
}

void BasicBlockNode::removeNode() {
  Parent->removeNode(this);
}

// TODO: Check why this implementation is really necessary.
void BasicBlockNode::printAsOperand(raw_ostream &O, bool PrintType) {
  O << Name;
}

void BasicBlockNode::removeSuccessor(BasicBlockNode *Successor) {
  revng_assert(not isCheck()); // TODO: maybe add removeTrue and removeFalse?
  for (auto It = Successors.begin(); It != Successors.end(); It++) {
    if (*It == Successor) {
      Successors.erase(It);
      break;
    }
  }
}

void BasicBlockNode::removePredecessor(BasicBlockNode *Predecessor) {
  for (auto It = Predecessors.begin(); It != Predecessors.end(); It++) {
    if (*It == Predecessor) {
      Predecessors.erase(It);
      break;
    }
  }
}


using BBNodeMap = std::map<BasicBlockNode *, BasicBlockNode *>;

static void handleNeighbors(const BBNodeMap &SubstitutionMap,
                             BasicBlockNode::links_container & Neighbors) {
  Neighbors.erase(std::remove_if(Neighbors.begin(),
                                 Neighbors.end(),
                                 [&SubstitutionMap](BasicBlockNode *N) {
                                   return SubstitutionMap.count(N) == 0;
                                 }),
                                 Neighbors.end());
  for (BasicBlockNode *&Neighbor : Neighbors) {
    revng_assert(SubstitutionMap.count(Neighbor) != 0);
    Neighbor = SubstitutionMap.at(Neighbor);
  }
}

void BasicBlockNode::updatePointers(const BBNodeMap &SubstitutionMap) {
  handleNeighbors(SubstitutionMap, Predecessors);
  handleNeighbors(SubstitutionMap, Successors);
}
