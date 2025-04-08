/// \file ScopeGraphAlgorithms.cpp
/// Helpers for the `ScopeGraph` building
///

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SetVector.h"
#include "llvm/IR/BasicBlock.h"

#include "revng/ADT/ReversePostOrderTraversal.h"
#include "revng/RestructureCFG/ScopeGraphAlgorithms.h"
#include "revng/RestructureCFG/ScopeGraphGraphTraits.h"

using namespace llvm;

SmallSetVector<BasicBlock *, 2> getScopeGraphSuccessors(BasicBlock *N) {
  // We employ a `SetVector` so that we do not take into account
  // multiplicity for edges out of a conditional
  SmallSetVector<BasicBlock *, 2> ConditionalSuccessors;
  for (BasicBlock *Successor : children<Scope<BasicBlock *>>(N)) {
    ConditionalSuccessors.insert(Successor);
  }

  return ConditionalSuccessors;
}

SmallSetVector<BasicBlock *, 2> getScopeGraphPredecessors(BasicBlock *N) {
  // It is important that we use a `SetVector` here in order to
  // deduplicate the successors outputted by the `llvm::children` range
  // iterator
  SmallSetVector<BasicBlock *, 2> Predecessors;
  for (auto *Predecessor : children<Inverse<Scope<BasicBlock *>>>(N)) {
    Predecessors.insert(Predecessor);
  }

  return Predecessors;
}

SmallVector<BasicBlock *> getNodesInScope(BasicBlock *ScopeEntryBlock,
                                          BasicBlock *PostDominator) {
  // We exploit the `Visited` set, by passing it to
  // `ReversePostOrderTraversalExt`, in order to stop the visit at the
  // `PostDominator`
  std::set<BasicBlock *> Visited;
  Visited.insert(PostDominator);

  // We collect all the nodes between the `Conditional` and its
  // immediate postdominator, by using the `ReversePostOrderTraversalExt`
  SmallVector<BasicBlock *> NodesToProcess;
  for (BasicBlock *RPONode :
       ReversePostOrderTraversalExt<Scope<BasicBlock *>>(ScopeEntryBlock,
                                                         Visited)) {
    NodesToProcess.push_back(RPONode);
  }

  // From the collected nodes, we need to remove the first node, which
  // corresponds to the `Conditional`, which should not be processed in this
  // round
  revng_assert(NodesToProcess.front() == ScopeEntryBlock);
  NodesToProcess.erase(NodesToProcess.begin());

  return NodesToProcess;
}
