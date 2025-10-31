//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/GenericDomTree.h"

#include "revng/RestructureCFG/ScopeGraphAlgorithms.h"
#include "revng/RestructureCFG/ScopeGraphGraphTraits.h"
#include "revng/RestructureCFG/ScopeGraphUtils.h"
#include "revng/RestructureCFG/SelectScopePass.h"
#include "revng/Support/Debug.h"
#include "revng/Support/GraphAlgorithms.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

// Debug logger
static Logger Log("select-scope");

static std::map<BasicBlock *, size_t>
initializeScopeMap(SmallSetVector<BasicBlock *, 2> &ConditionalSuccessors) {
  std::map<BasicBlock *, size_t> ScopeMap;

  // We initialize the `ScopeMap` for each `ConditionalSuccessor`
  for (const auto &[Index, ConditionalSuccessor] :
       enumerate(ConditionalSuccessors)) {
    if (not ScopeMap.contains(ConditionalSuccessor)) {
      ScopeMap[ConditionalSuccessor] = Index;
    }
  }
  return ScopeMap;
}

/// Helper function used to elect the `ScopeID` for a node, electing the
/// `ScopeID` for which we have the highest number of predecessors having such
/// `ScopeID`
static size_t electMaxScopeID(SmallSetVector<BasicBlock *, 2> &Predecessors,
                              std::map<BasicBlock *, size_t> &ScopeMap) {
  // In the map, we store the how frequent a `ScopeID` is in the
  // predecessors of `Candidate`, in order to elect the one with the most
  // common `ScopeID`
  DenseMap<size_t, size_t> PredecessorsScopeCounter;
  for (auto *Predecessor : Predecessors) {
    auto ScopeMapIt = ScopeMap.find(Predecessor);
    if (ScopeMapIt != ScopeMap.end()) {
      size_t PredecessorScopeID = ScopeMapIt->second;
      PredecessorsScopeCounter[PredecessorScopeID]++;
    }
  }

  // We must have found at least a predecessor with an assigned `ScopeID`
  revng_assert(PredecessorsScopeCounter.size() > 0);

  // We elect the `ScopeID` for which we have the maximum number of predecessors
  // having such `ScopeID`
  size_t MaxOccurencies = PredecessorsScopeCounter.begin()->second;
  size_t MaxScopeID = PredecessorsScopeCounter.begin()->first;
  for (const auto &[Key, Value] : PredecessorsScopeCounter) {
    if (Value > MaxOccurencies) {
      MaxOccurencies = Value;
      MaxScopeID = Key;
    }
  }

  return MaxScopeID;
}

class SelectScopePassImpl {
  Function &F;
  PostDomTreeOnView<BasicBlock, Scope> PDT;
  const ScopeGraphBuilder SGBuilder;

  // We keep a boolean field to track whether the `Function` was modified
  bool FunctionModified = false;

public:
  SelectScopePassImpl(Function &F) : F(F), SGBuilder(&F) {}

public:
  void processConditionalNode(BasicBlock *ConditionalNode) {
    SmallSetVector<BasicBlock *, 2>
      ConditionalSuccessors = getScopeGraphSuccessors(ConditionalNode);

    // We skip all the nodes which are not conditional
    if (ConditionalSuccessors.size() <= 1) {
      return;
    }

    revng_log(Log,
              "Processing conditional " << ConditionalNode->getName().str()
                                        << "\n");

    BasicBlock *PostDominator = PDT[ConditionalNode]->getIDom()->getBlock();
    revng_assert(PostDominator);
    revng_log(Log,
              "The identified postdominator is "
                << PostDominator->getName().str() << "\n");

    SmallVector<BasicBlock *> NodesToProcess = getNodesInScope(ConditionalNode,
                                                               PostDominator);
    revng_log(Log,
              "Nodes between conditional and its postdominator, in reverse "
              "post order:\n");
    for (auto DFSNode : NodesToProcess) {
      revng_log(Log, "  " << DFSNode->getName().str());
    }

    // Initialize the `ScopeMap`
    std::map<BasicBlock *, size_t>
      ScopeMap = initializeScopeMap(ConditionalSuccessors);

    // Process each node in the zone of interest
    for (BasicBlock *Candidate : NodesToProcess) {
      revng_log(Log,
                "Analyzing candidate: " + Candidate->getName().str() << "\n");

      // We precompute the predecessors to avoid invalidation due to graph
      // changes. It is fundamental that we always traverse the `ScopeGraph`
      // view of the CFG, or we may end up with some inconsistencies in terms
      // of the visited nodes.
      revng_log(Log, "The candidate predecessors are:\n");
      SmallSetVector<BasicBlock *, 2>
        Predecessors = getScopeGraphPredecessors(Candidate);

      if (Log.isEnabled()) {
        for (BasicBlock *Predecessor : Predecessors) {
          Log << "  " << Predecessor->getName() << "\n";
        }
      }

      // `Candidate`, could be itself a immediate successor of a conditional
      // node, and therefore correspond to a `ScopeID`. We therefore need to
      // take into consideration it when assigning the final scope for each
      // `Candidate`.
      // We can do this by always enqueuing `Candidate` as a predecessor of
      // itself, this can lead to two situations:
      // 1) `Candidate` is not a successor of the conditional, therefore no
      //    corresponding entry in `ScopeMap` will be present, and this
      //    will not influence the decision on the `ScopeID` which will be
      //    finally assigned.
      // 2) `Candidate` is a successor of the conditional, therefore a
      //    corresponding entry in `ScopeMap` will be present, and it
      //    will be correctly taken into account for the `ScopeID` decision
      //    process.
      // Alternatively, we could pre-assign the `ScopeID`, in the
      // `ScopeMap`, for each successor of a conditional node during
      // the initialization. This, however, would tie us to the decision of
      // always assigning the successor of a conditional node to the `ScopeID`
      // opening in the successor itself, while, in principle, we could
      // alternatively disconnect the edge connecting the conditional and the
      // successor, by making it a `goto` edge.
      Predecessors.insert(Candidate);

      std::optional<size_t> ElectedScopeID;

      // If we already assigned the `ScopeID` for the current node, we maintain
      // that one. This is necessary for the `Candidate`s that are successors of
      // the `ConditionalNode` itself, for which the decision is mandatory
      // (being assigned to the `ScopeID` they themselves open).
      auto CandidateScopeMapIt = ScopeMap.find(Candidate);
      if (CandidateScopeMapIt != ScopeMap.end()) {
        ElectedScopeID = CandidateScopeMapIt->second;
      } else {

        // We elect the `ScopeID` for `Candidate` electing the `ScopeID` for
        // which we have the maximum number of predecessors having a certain
        // `ScopeID`
        ElectedScopeID = electMaxScopeID(Predecessors, ScopeMap);
      }
      revng_assert(ElectedScopeID);

      for (auto *Predecessor : Predecessors) {
        auto ScopeMapIt = ScopeMap.find(Predecessor);

        // We may have two situations: 1) There is an entry for `Predecessor`
        // in the `ScopeMap`, it means that there is a path connecting
        // the `Conditional` and `Candidate`. 2) There is no entry for
        // `Predecessor`, therefore such node wasn't visited during the
        // current exploration of the zone of interest, and therefore it does
        // not lie on any path between the `Conditional` and the
        // `Candidate` node.
        if (ScopeMapIt != ScopeMap.end()) {
          size_t PredecessorScopeID = ScopeMapIt->second;

          // If the `PredecessorScopeID` is different from the one already
          // assigned to `Candidate`, we need to transform the edge into a
          // `goto` edge, in order to respect the decidedness definition.
          if (PredecessorScopeID != ElectedScopeID) {
            SGBuilder.makeGotoEdge(Predecessor, Candidate);

            revng_log(Log,
                      "Inserting a goto edge between predecessor "
                        << Predecessor->getName() << " -> and candidate "
                        << Candidate->getName());

            // We mark the CFG as modified
            FunctionModified = true;
          }
        }
      }

      // We insert in the `ScopeMap` a new entry once we assigned the
      // final `ScopeID` to `Candidate`, reflecting the final `ScopeID` that
      // we elected in the above process
      ScopeMap[Candidate] = *ElectedScopeID;
    }
  }

  bool run() {
    Scope<Function *> ScopeGraph(&F);

    // We compute the `PostDominatorTree` at the beginning of the pass, and we
    // do not update it, as per design, in order not to take into consideration
    // the changing PDT (changes caused by insertion of new exit nodes,
    // represented by the `goto` blocks)
    PDT.recalculate(F);

    // We iterate over the conditional nodes in the `ScopeGraph` in post order,
    // and we apply the `SelectScope` transformation for each conditional
    for (BasicBlock *ConditionalNode : post_order(ScopeGraph)) {
      processConditionalNode(ConditionalNode);
    }

    // We verify that the `ScopeGraph` has not blocks disconnected from the
    // entry block
    if (VerifyLog.isEnabled()) {
      revng_assert(not hasUnreachableBlocks(&F));
    }

    return FunctionModified;
  }
};

char SelectScopePass::ID = 0;
static constexpr const char *Flag = "select-scope";
using Reg = llvm::RegisterPass<SelectScopePass>;
static Reg X(Flag, "Perform the SelectScope pass on the ScopeGraph");

bool SelectScopePass::runOnFunction(llvm::Function &F) {
  // Log the function name
  revng_log(Log, "Running SelectScope on function " << F.getName() << "\n");

  // Instantiate and call the `Impl` class
  SelectScopePassImpl SelectScopeImpl(F);
  bool FunctionModified = SelectScopeImpl.run();

  // This pass may change the CFG by transforming some edges into `goto` edges,
  // therefore creating some additional `goto_block`s. We propagate the
  // information computed by the `Impl` class.
  return FunctionModified;
}

void SelectScopePass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  // This pass does not preserve the CFG
}
