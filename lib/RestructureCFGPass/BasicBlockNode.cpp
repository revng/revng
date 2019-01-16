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
#include "revng-c/RestructureCFGPass/Utils.h"

using namespace llvm;

// Forward declaration.
class CFG;

BasicBlockNode::BasicBlockNode(BasicBlock *BB, CFG *Parent) :
  BB(BB),
  IsReturn(false),
  Name(BB->getName()),
  Parent(Parent),
  Dummy(false),
  CollapsedRegion(nullptr) {}

BasicBlockNode::BasicBlockNode(std::string Name, CFG *Parent, bool IsDummy) :
  BB(nullptr),
  IsReturn(false),
  Name(Name),
  Parent(Parent),
  Dummy(IsDummy),
  CollapsedRegion(nullptr) {}

bool BasicBlockNode::isReturn() { return IsReturn; }
void BasicBlockNode::setReturn() { IsReturn = true; }

CFG *BasicBlockNode::getParent() {
  revng_assert(Parent != nullptr);
  return Parent;
}

bool BasicBlockNode::isDummy() {
  return Dummy;
}

// TODO: Check why this implementation is really necessary.
void BasicBlockNode::printAsOperand(raw_ostream &O, bool PrintType) {
  O << Name;
}

void BasicBlockNode::addSuccessor(BasicBlockNode *Successor) {
  Successors.push_back(Successor);
}

void BasicBlockNode::removeSuccessor(BasicBlockNode *Successor) {
  for (auto It = Successors.begin(); It != Successors.end(); It++) {
    if (*It == Successor) {
      Successors.erase(It);
      break;
    }
  }
}

void BasicBlockNode::addPredecessor(BasicBlockNode *Predecessor) {
  Predecessors.push_back(Predecessor);
}

void BasicBlockNode::removePredecessor(BasicBlockNode *Predecessor) {
  for (auto It = Predecessors.begin(); It != Predecessors.end(); It++) {
    if (*It == Predecessor) {
      Predecessors.erase(It);
      break;
    }
  }
}

void BasicBlockNode::updatePointers(std::map<BasicBlockNode *,
                    BasicBlockNode *> &SubstitutionMap) {

  // Handle predecessors.
  Predecessors.erase(std::remove_if(Predecessors.begin(),
                                    Predecessors.end(),
                                    [&SubstitutionMap](BasicBlockNode *N)
                                    { return
                                        SubstitutionMap.count(N) == 0; }),
                                    Predecessors.end());

  for (auto It = Predecessors.begin(); It != Predecessors.end(); It++) {
    revng_assert(SubstitutionMap.count(*It) != 0);
    (*It) = SubstitutionMap[*It];
  }

  // Handle successors.
  Successors.erase(std::remove_if(Successors.begin(),
                                  Successors.end(),
                                  [&SubstitutionMap](BasicBlockNode *N)
                                  { return SubstitutionMap.count(N) == 0; }),
                                  Successors.end());

  for (auto It = Successors.begin(); It != Successors.end(); It++) {
    revng_assert(SubstitutionMap.count(*It) != 0);
    (*It) = SubstitutionMap[*It];
  }

}

BasicBlockNode *BasicBlockNode::getPredecessorI(size_t i) {
  return Predecessors[i];
}

BasicBlockNode *BasicBlockNode::getSuccessorI(size_t i) {
  return Successors[i];
}


BasicBlock *BasicBlockNode::basicBlock() const { return BB; }
StringRef BasicBlockNode::getName() const { return Name; }
std::string BasicBlockNode::getNameStr() const { return Name; }

void BasicBlockNode::setCollapsedCFG(CFG *Graph) {
  CollapsedRegion = Graph;
}

bool BasicBlockNode::isCollapsed() {
  return CollapsedRegion != nullptr;
}

CFG *BasicBlockNode::getCollapsedCFG() {
  return CollapsedRegion;
}

// EdgeDescriptor is a handy way to create and manipulate edges on the CFG.
using EdgeDescriptor = std::pair<BasicBlockNode *, BasicBlockNode *>;

static bool edgesEqual(EdgeDescriptor &First, EdgeDescriptor &Second) {
  if ((First.first == Second.first) and (First.second == Second.second)) {
    return true;
  } else {
    return false;
  }
}

bool containsEdge(std::set<EdgeDescriptor> &Container, EdgeDescriptor &Edge) {
  for (EdgeDescriptor Elem : Container) {
    if (edgesEqual(Elem, Edge)) {
      return true;
    }
  }
  return false;
}
