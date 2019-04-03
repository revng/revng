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
                               RegionCFG *Collapsed,
                               StringRef Name,
                               Type T,
                               unsigned Value) :
  ID(Parent->getNewID()),
  Parent(Parent),
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
  // TODO: maybe add removeTrue and removeFalse?
  // revng_assert(not isCheck());
  size_t Removed = 0;
  Successors.erase(std::remove_if(Successors.begin(),
                                  Successors.end(),
                                  [&Removed, Successor](BasicBlockNode *B) {
                                    if (B == Successor) {
                                      Removed++;
                                      return true;
                                    }
                                    return false;
                                  }),
                    Successors.end());
  revng_assert(Removed == 1); // needs to remove exactly one successor
}

void BasicBlockNode::removePredecessor(BasicBlockNode *Predecessor) {
  size_t Removed = 0;
  Predecessors.erase(std::remove_if(Predecessors.begin(),
                                   Predecessors.end(),
                                   [&Removed, Predecessor](BasicBlockNode *B) {
                                     if (B == Predecessor) {
                                       Removed++;
                                       return true;
                                     }
                                     return false;
                                   }),
                     Predecessors.end());
  revng_assert(Removed == 1); // needs to remove exactly one predecessor
}

using BBNodeMap = std::map<BasicBlockNode *, BasicBlockNode *>;

static void handleNeighbors(const BBNodeMap &SubstitutionMap,
                            BasicBlockNode::links_container &Neighbors) {
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
  if (not isCheck()) {
    handleNeighbors(SubstitutionMap, Successors);
  } else { // don't screw up the if/then/else of check nodes

    auto It = SubstitutionMap.find(getTrue());
    if (It != SubstitutionMap.end())
      setTrue(It->second);
    else
      setTrue(nullptr);

    It = SubstitutionMap.find(getFalse());
    if (It != SubstitutionMap.end())
      setFalse(It->second);
    else
      setFalse(nullptr);
  }
}

StringRef BasicBlockNode::getName() const {
  switch (NodeType) {
    case Type::Code:
      return Name;
    case Type::Empty:
      return "dummy";
    case Type::Break:
      return "break";
    case Type::Continue:
      return "continue";
    case Type::Set:
      return "set " + getStateVariableValue();
    case Type::Check:
      return "check " + getStateVariableValue();
    case Type::Collapsed:
      return "collapsed";
    default:
      revng_abort("Artificial node not expected");
  }
}

bool BasicBlockNode::isEquivalentTo(BasicBlockNode *Other) {

  // TODO: this algorithm fails if there are nodes in the graph not reachable
  //       from the entry node (even if the number of nodes in the two
  //       `RegionCFG` is equal).

  // Early failure if the IDs of the nodes are different.
  if (getID() != Other->getID()) {
    return false;
  }

  // Early failure if the number of successors for a node is node equal.
  size_t SuccessorNumber = successor_size();
  size_t OtherSuccessorNumber = Other->successor_size();
  if (SuccessorNumber != OtherSuccessorNumber) {
    return false;
  }

  bool ComparisonState = true;
  for (size_t I = 0; I < SuccessorNumber; I++) {
    BasicBlockNode *SuccessorI = getSuccessorI(I);
    BasicBlockNode *OtherSuccessorI = Other->getSuccessorI(I);
    ComparisonState &= SuccessorI->isEquivalentTo(OtherSuccessorI);
  }

  return ComparisonState;

}
