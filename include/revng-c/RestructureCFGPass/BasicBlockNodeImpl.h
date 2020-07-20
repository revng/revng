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
#include <revng-c/RestructureCFGPass/BasicBlockNode.h>
#include <revng-c/RestructureCFGPass/RegionCFGTreeBB.h>
#include <revng-c/RestructureCFGPass/Utils.h>

// Trait exposing the weight of a generic object wrapped by `BasicBlockNode`.
template<class T>
struct WeightTraits {};

// Specialization of the WeightTraits for the the `BasicBlock` class, which
// simply returns the number of instruction composing it.
template<>
struct WeightTraits<llvm::BasicBlock *> {

  // The `-1` is added to discard the dimension of blocks composed only by a
  // branching instruction (which we may insert, but virtually add no extra
  // weight.
  static size_t getWeight(llvm::BasicBlock *BB) { return BB->size() - 1; }
};

template<class NodeT>
inline BasicBlockNode<NodeT>::BasicBlockNode(RegionCFGT *Parent,
                                             NodeT OriginalNode,
                                             RegionCFGT *Collapsed,
                                             llvm::StringRef Name,
                                             Type T,
                                             unsigned Value) :
  ID(Parent->getNewID()),
  Parent(Parent),
  CollapsedRegion(Collapsed),
  NodeType(T),
  Name(Name),
  StateVariableValue(Value),
  OriginalNode(OriginalNode),
  Weaved(false) {
}

// Needed by `DomTreeBuilder`.
template<class NodeT>
inline void BasicBlockNode<NodeT>::printAsOperand(llvm::raw_ostream &O,
                                                  bool /* PrintType */) const {
  O << Name;
}

template<class NodeT>
inline void BasicBlockNode<NodeT>::removeSuccessor(BasicBlockNodeT *Succ) {
  size_t Removed = 0;
  const auto IsSucc = [&Removed, Succ](const node_label_pair &P) {
    if (P.first == Succ) {
      ++Removed;
      return true;
    }
    return false;
  };
  Successors.erase(std::remove_if(Successors.begin(), Successors.end(), IsSucc),
                   Successors.end());
  revng_assert(Removed == 1); // needs to remove exactly one successor
}

template<class NodeT>
inline typename BasicBlockNode<NodeT>::node_label_pair
BasicBlockNode<NodeT>::extractSuccessorEdge(BasicBlockNodeT *Succ) {
  size_t Removed = 0;
  node_label_pair Extracted;
  const auto IsSucc = [&Removed, &Extracted, Succ](const node_label_pair &P) {
    if (P.first == Succ) {
      ++Removed;
      Extracted = P;
      return true;
    }
    return false;
  };
  Successors.erase(std::remove_if(Successors.begin(), Successors.end(), IsSucc),
                   Successors.end());
  revng_assert(Removed == 1); // needs to remove exactly one successor
  return Extracted;
}

template<class NodeT>
inline void BasicBlockNode<NodeT>::removePredecessor(BasicBlockNodeT *Pred) {
  size_t Removed = 0;
  const auto IsPred = [&Removed, Pred](const node_label_pair &P) {
    if (P.first == Pred) {
      ++Removed;
      return true;
    }
    return false;
  };
  Predecessors.erase(std::remove_if(Predecessors.begin(),
                                    Predecessors.end(),
                                    IsPred),
                     Predecessors.end());
  revng_assert(Removed == 1); // needs to remove exactly one predecessor
}

template<class NodeT>
inline typename BasicBlockNode<NodeT>::node_label_pair
BasicBlockNode<NodeT>::extractPredecessorEdge(BasicBlockNodeT *Pred) {
  size_t Removed = 0;
  node_label_pair Extracted;
  const auto IsPred = [&Removed, &Extracted, Pred](const node_label_pair &P) {
    if (P.first == Pred) {
      ++Removed;
      Extracted = P;
      return true;
    }
    return false;
  };
  Predecessors.erase(std::remove_if(Predecessors.begin(),
                                    Predecessors.end(),
                                    IsPred),
                     Predecessors.end());
  revng_assert(Removed == 1); // needs to remove exactly one predecessor
  return Extracted;
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
                                 [&SubMap](const auto &NodeWithLabels) {
                                   return !SubMap.count(NodeWithLabels.first);
                                 }),
                  Neighbors.end());

  for (auto &NeighborLabelPair : Neighbors) {
    auto &Neighbor = NeighborLabelPair.first;
    revng_assert(SubMap.count(Neighbor) != 0);
    Neighbor = SubMap.at(Neighbor);
  }
}

template<class NodeT>
inline void BasicBlockNode<NodeT>::updatePointers(const BBNodeMap &SubMap) {
  handleNeighbors<NodeT>(SubMap, Predecessors);
  handleNeighbors<NodeT>(SubMap, Successors);
}

template<class NodeT>
inline llvm::StringRef BasicBlockNode<NodeT>::getName() const {
  return llvm::StringRef(Name);
}

template<class NodeT>
inline bool
BasicBlockNode<NodeT>::isEquivalentTo(BasicBlockNodeT *Other) const {

  // Early failure if the IDs of the nodes are different.
  if (getID() != Other->getID()) {
    return false;
  }

  // Early failure if the number of successors for a node is not equal.
  size_t SuccessorNumber = successor_size();
  size_t OtherSuccessorNumber = Other->successor_size();
  if (SuccessorNumber != OtherSuccessorNumber) {
    return false;
  }

  for (size_t I = 0; I < SuccessorNumber; I++) {
    BasicBlockNode *SuccessorI = getSuccessorI(I);
    BasicBlockNode *OtherSuccessorI = Other->getSuccessorI(I);
    if (not SuccessorI->isEquivalentTo(OtherSuccessorI)) {
      return false;
    }
  }

  return true;
}

template<class NodeT>
inline size_t BasicBlockNode<NodeT>::getWeight() const {
  if (NodeType == Type::Code) {
    revng_assert(OriginalNode != nullptr);
    return WeightTraits<NodeT>::getWeight(OriginalNode);
  } else if (NodeType == Type::Empty) {
    return 0;
  } else if (NodeType == Type::Break) {
    return 0;
  } else if (NodeType == Type::Continue) {
    return 0;
  } else if (NodeType == Type::Set) {
    return 0;
  } else if (NodeType == Type::Dispatcher) {
    return 0;
  } else if (NodeType == Type::Collapsed) {
    revng_assert(CollapsedRegion != nullptr);
    size_t WeightAccumulator = 0;
    for (BasicBlockNode<NodeT> *CollapsedNode : CollapsedRegion->nodes()) {
      WeightAccumulator += CollapsedNode->getWeight();
    }
    return WeightAccumulator;
  } else {
    revng_abort("getWeight() still not implemented.");
  }
}

#endif // REVNGC_RESTRUCTURE_CFG_BASICBLOCKNODEIMPL_H
