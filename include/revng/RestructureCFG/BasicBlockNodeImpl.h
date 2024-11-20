#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdlib>
#include <map>
#include <set>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"

#include "revng/RestructureCFG/BasicBlockNode.h"
#include "revng/RestructureCFG/RegionCFGTreeBB.h"
#include "revng/RestructureCFG/Utils.h"

// Trait exposing the weight of a generic object wrapped by `BasicBlockNode`.
template<class T>
struct WeightTraits {};

// Specialization of the WeightTraits for the the `BasicBlock` class, which
// simply returns the number of instruction composing it.
template<>
struct WeightTraits<llvm::BasicBlock *> {

  static size_t getWeight(llvm::BasicBlock *BB) {
    // By default they weight of a BB is the number of instructions it contains.
    size_t Weight = BB->size();
    // If the terminator is an unconditional branch we decrease the weight by
    // one, because unconditional branches are never emitted in C.
    if (auto *Br = dyn_cast<llvm::BranchInst>(BB->getTerminator())) {
      if (Br->isUnconditional())
        --Weight;
    }
    return Weight;
  }
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
  const auto IsSucc = [&Removed, Succ](const node_edgeinfo_pair &P) {
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
inline typename BasicBlockNode<NodeT>::node_edgeinfo_pair
BasicBlockNode<NodeT>::extractSuccessorEdge(BasicBlockNodeT *Succ) {
  size_t Removed = 0;
  node_edgeinfo_pair Extracted;
  const auto IsSucc =
    [&Removed, &Extracted, Succ](const node_edgeinfo_pair &P) {
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
inline const typename BasicBlockNode<NodeT>::node_edgeinfo_pair &
BasicBlockNode<NodeT>::getSuccessorEdge(const BasicBlockNodeT *Succ) const {
  const auto IsSucc = [Succ](const node_edgeinfo_pair &P) {
    return P.first == Succ;
  };
  auto SuccEnd = Successors.end();
  auto SuccEdgeIt = std::find_if(Successors.begin(), SuccEnd, IsSucc);
  revng_assert(SuccEdgeIt != SuccEnd);
  revng_assert(std::find_if(std::next(SuccEdgeIt), SuccEnd, IsSucc) == SuccEnd);
  return *SuccEdgeIt;
}

template<class NodeT>
inline typename BasicBlockNode<NodeT>::node_edgeinfo_pair &
BasicBlockNode<NodeT>::getSuccessorEdge(BasicBlockNodeT *Succ) {
  const auto IsSucc = [Succ](node_edgeinfo_pair &P) { return P.first == Succ; };
  auto SuccEnd = Successors.end();
  auto SuccEdgeIt = std::find_if(Successors.begin(), SuccEnd, IsSucc);
  revng_assert(SuccEdgeIt != SuccEnd);
  revng_assert(std::find_if(std::next(SuccEdgeIt), SuccEnd, IsSucc) == SuccEnd);
  return *SuccEdgeIt;
}

template<class NodeT>
inline void BasicBlockNode<NodeT>::removePredecessor(BasicBlockNodeT *Pred) {
  size_t Removed = 0;
  const auto IsPred = [&Removed, Pred](const node_edgeinfo_pair &P) {
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
inline typename BasicBlockNode<NodeT>::node_edgeinfo_pair
BasicBlockNode<NodeT>::extractPredecessorEdge(BasicBlockNodeT *Pred) {
  size_t Removed = 0;
  node_edgeinfo_pair Extracted;
  const auto IsPred =
    [&Removed, &Extracted, Pred](const node_edgeinfo_pair &P) {
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
inline const typename BasicBlockNode<NodeT>::node_edgeinfo_pair &
BasicBlockNode<NodeT>::getPredecessorEdge(const BasicBlockNodeT *Pred) const {
  const auto IsPred = [Pred](const node_edgeinfo_pair &P) {
    return P.first == Pred;
  };
  auto PredEnd = Predecessors.end();
  auto PredEdgeIt = std::find_if(Predecessors.begin(), PredEnd, IsPred);
  revng_assert(PredEdgeIt != PredEnd);
  revng_assert(std::find_if(std::next(PredEdgeIt), PredEnd, IsPred) == PredEnd);
  return *PredEdgeIt;
}

template<class NodeT>
inline typename BasicBlockNode<NodeT>::node_edgeinfo_pair &
BasicBlockNode<NodeT>::getPredecessorEdge(BasicBlockNodeT *Pred) {
  const auto IsPred = [Pred](node_edgeinfo_pair &P) { return P.first == Pred; };
  auto PredEnd = Predecessors.end();
  auto PredEdgeIt = std::find_if(Predecessors.begin(), PredEnd, IsPred);
  revng_assert(PredEdgeIt != PredEnd);
  revng_assert(std::find_if(std::next(PredEdgeIt), PredEnd, IsPred) == PredEnd);
  return *PredEdgeIt;
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
                                 [&SubMap](const auto &LabeledNode) {
                                   return !SubMap.contains(LabeledNode.first);
                                 }),
                  Neighbors.end());

  for (auto &NeighborLabelPair : Neighbors) {
    auto &Neighbor = NeighborLabelPair.first;
    revng_assert(SubMap.contains(Neighbor));
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

  switch (NodeType) {
  case Type::Code: {
    revng_assert(OriginalNode != nullptr);
    return WeightTraits<NodeT>::getWeight(OriginalNode);
  } break;

  case Type::Collapsed: {
    revng_assert(CollapsedRegion != nullptr);
    return CollapsedRegion->getUntangleWeight();
  } break;

  case Type::Tile: {
    revng_abort("getWeight() not implemented for tiles.");
  } break;

  case Type::Empty: {
    // Empty nodes contain nothing, their weight is 0
    return 0;
  } break;

  case Type::EntryDispatcher:
  case Type::ExitDispatcher:
  case Type::Break:
  case Type::Continue:
  case Type::EntrySet:
  case Type::ExitSet: {
    // These nodes all cost 1, because they contain a single statement.
    return 1;
  } break;

  default:
    revng_abort("getWeight() still not implemented.");
  }

  return 0;
}
