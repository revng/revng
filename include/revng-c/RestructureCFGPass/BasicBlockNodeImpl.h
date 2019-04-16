#ifndef REVNGC_RESTRUCTURE_CFG_BASICBLOCKNODEIMPL_H
#define REVNGC_RESTRUCTURE_CFG_BASICBLOCKNODEIMPL_H

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

template<class NodeT>
inline BasicBlockNode<NodeT>::BasicBlockNode(RegionCFGT *Parent,
                                             NodeT OriginalNode,
                                             RegionCFGT *Collapsed,
                                             llvm::StringRef Name,
                                             Type T,
                                             unsigned Value) :
  ID(Parent->getNewID()),
  Parent(Parent),
  OriginalNode(OriginalNode),
  CollapsedRegion(Collapsed),
  NodeType(T),
  Name(Name),
  StateVariableValue(Value) {
}

template<class NodeT>
inline void BasicBlockNode<NodeT>::removeNode() {
  Parent->removeNode(this);
}

template<class NodeT>
// TODO: Check why this implementation is really necessary.
inline void BasicBlockNode<NodeT>::printAsOperand(llvm::raw_ostream &O,
                                                  bool PrintType) const {
  O << Name;
}

template<class NodeT>
inline void BasicBlockNode<NodeT>::removeSuccessor(BasicBlockNodeT *Successor) {
  // TODO: maybe add removeTrue and removeFalse?
  // revng_assert(not isCheck());
  size_t Removed = 0;
  Successors.erase(std::remove_if(Successors.begin(),
                                  Successors.end(),
                                  [&Removed, Successor](BasicBlockNodeT *B) {
                                    if (B == Successor) {
                                      Removed++;
                                      return true;
                                    }
                                    return false;
                                  }),
                    Successors.end());
  revng_assert(Removed == 1); // needs to remove exactly one successor
}

template<class NodeT>
inline void BasicBlockNode<NodeT>::removePredecessor(BasicBlockNodeT *Pred) {
  size_t Removed = 0;
  Predecessors.erase(std::remove_if(Predecessors.begin(),
                                   Predecessors.end(),
                                   [&Removed, Pred](BasicBlockNodeT *B) {
                                     if (B == Pred) {
                                       Removed++;
                                       return true;
                                     }
                                     return false;
                                   }),
                     Predecessors.end());
  revng_assert(Removed == 1); // needs to remove exactly one predecessor
}

template<class NodeT>
using BBNodeMap = typename BasicBlockNode<NodeT>::BBNodeMap;

template<class NodeT>
using links_container = typename BasicBlockNode<NodeT>::links_container;

template<class NodeT>
inline void handleNeighbors(const BBNodeMap<NodeT> &SubMap,
                            links_container<NodeT> &Neighbors) {
  Neighbors.erase(std::remove_if(Neighbors.begin(),
                                 Neighbors.end(),
                                 [&SubMap](BasicBlockNode<NodeT> *N) {
                                   return SubMap.count(N) == 0;
                                 }),
                  Neighbors.end());
  for (BasicBlockNode<NodeT> *&Neighbor : Neighbors) {
    revng_assert(SubMap.count(Neighbor) != 0);
    Neighbor = SubMap.at(Neighbor);
  }
}

template<class NodeT>
inline void BasicBlockNode<NodeT>::updatePointers(const BBNodeMap &SubMap) {
  handleNeighbors<NodeT>(SubMap, Predecessors);
  if (not isCheck()) {
    handleNeighbors<NodeT>(SubMap, Successors);
  } else { // don't screw up the if/then/else of check nodes

    auto It = SubMap.find(getTrue());
    if (It != SubMap.end())
      setTrue(It->second);
    else
      setTrue(nullptr);

    It = SubMap.find(getFalse());
    if (It != SubMap.end())
      setFalse(It->second);
    else
      setFalse(nullptr);
  }
}

template<class NodeT>
inline llvm::StringRef BasicBlockNode<NodeT>::getName() const {
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
      return "set " + std::to_string(getStateVariableValue());
    case Type::Check:
      return "check " + std::to_string(getStateVariableValue());
    case Type::Collapsed:
      return "collapsed";
    default:
      revng_abort("Artificial node not expected");
  }
}

template<class NodeT>
inline bool
BasicBlockNode<NodeT>::isEquivalentTo(BasicBlockNodeT *Other) const {

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

#endif // REVNGC_RESTRUCTURE_CFG_BASICBLOCKNODEIMPL_H
