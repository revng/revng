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

bool isScopeGraphDecided(Function &F) {
  Scope<Function *> ScopeGraph(&F);

  // We compute the `DominatorTree` and the `PostDominatorTree` on the
  // `ScopeGraph`
  DomTreeOnView<BasicBlock, Scope> DT;
  PostDomTreeOnView<BasicBlock, Scope> PDT;
  DT.recalculate(F);
  PDT.recalculate(F);

  // We iterate over the conditional nodes in the `ScopeGraph` in post order,
  // and we check for the decidedness
  for (BasicBlock *ConditionalNode : post_order(ScopeGraph)) {
    SmallSetVector<BasicBlock *, 2>
      ConditionalSuccessors = getScopeGraphSuccessors(ConditionalNode);

    // We skip all the nodes which are not conditional
    if (ConditionalSuccessors.size() <= 1) {
      continue;
    }

    // Collect all the nodes in the zone of interest of each `Successor` of a
    // `ConditionalNode`, i.e., all the nodes between the `Successor` and the
    // immediate `PostDominator` of `ConditionalNode`
    BasicBlock *PostDominator = PDT[ConditionalNode]->getIDom()->getBlock();
    for (auto *Successor : ConditionalSuccessors) {

      // If the `Successor` coincides with the `PostDominator`, we do not have
      // to check anything
      if (Successor == PostDominator) {
        continue;
      }

      SmallVector<BasicBlock *> NodesToProcess = getNodesInScope(Successor,
                                                                 PostDominator);

      // If we find a `Candidate` which is not dominated by the `Successor`,
      // it means the `ScopeGraph` has become undecided
      for (BasicBlock *Candidate : NodesToProcess) {
        if (not DT.dominates(Successor, Candidate)) {
          return false;
        }
      }
    }
  }

  return true;
}
