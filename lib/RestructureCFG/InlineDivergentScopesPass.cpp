//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/GenericDomTree.h"

#include "revng/RestructureCFG/InlineDivergentScopesPass.h"
#include "revng/RestructureCFG/ScopeGraphGraphTraits.h"
#include "revng/Support/Assert.h"
#include "revng/Support/GraphAlgorithms.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

/// Manager class to automatically handle the creation and clean up of the
/// `PlaceHolderTarget`
class PlaceHolderTargetManager {
private:
  BasicBlock *PlaceHolderTarget;

public:
  PlaceHolderTargetManager(Function &F) {
    LLVMContext &Context = getContext(&F);
    PlaceHolderTarget = BasicBlock::Create(Context,
                                           "placeholder_destination",
                                           &F);
  }

  ~PlaceHolderTargetManager() { PlaceHolderTarget->eraseFromParent(); }

  BasicBlock *operator*() const { return PlaceHolderTarget; }
};

static bool isConditional(BasicBlock *Node) {
  auto Successors = children<Scope<BasicBlock *>>(Node);
  size_t NumSuccessors = std::distance(Successors.begin(), Successors.end());
  if (NumSuccessors >= 2) {
    return true;
  } else {
    return false;
  }
}

/// Helper function which detects
static bool isExit(BasicBlock *Node) {
  auto Successors = children<Scope<BasicBlock *>>(Node);
  return std::ranges::empty(Successors);
}

/// Helper function to collect all the exit nodes on the `ScopeGraph`
static SmallVector<BasicBlock *> getExits(Function &F,
                                          BasicBlock *PlaceHolderTarget) {
  SmallVector<BasicBlock *> Exits;
  Scope<Function *> ScopeGraph(&F);
  for (BasicBlock *BB : nodes(ScopeGraph)) {

    // We need to exclude the `PlaceHolderTarget` which is a temporary block
    // which incidentally happens to be an exit, but will be removed before the
    // end of the pass
    if (BB != PlaceHolderTarget and isExit(BB)) {
      Exits.push_back(BB);
    }
  }

  return Exits;
}

/// Helper to obtain the immediate dominator `BasicBlock`, if present
static BasicBlock *
getImmediateDominator(BasicBlock *N,
                      DomTreeOnView<BasicBlock, Scope> &DomTree) {
  auto *Node = DomTree.getNode(N)->getIDom();
  if (Node) {
    return Node->getBlock();
  } else {
    return nullptr;
  }
}

/// Helper function which simplifies all the terminators containing
/// `PlaceHolderTarget`, by removing it
static void simplifyTerminator(BasicBlock *BB,
                               const BasicBlock *PlaceHolderTarget) {
  Instruction *Terminator = BB->getTerminator();

  if (auto *Branch = dyn_cast<BranchInst>(Terminator)) {
    if (Branch->isConditional()) {

      // We want to transform a conditional branch with one of the destination
      // set to `PlaceHolderTarget` to a non conditional branch
      BasicBlock *SingleDestination = nullptr;

      if (Branch->getSuccessor(0) == PlaceHolderTarget) {
        SingleDestination = Branch->getSuccessor(1);
        revng_assert(SingleDestination != PlaceHolderTarget);
      } else if (Branch->getSuccessor(1) == PlaceHolderTarget) {
        SingleDestination = Branch->getSuccessor(0);
        revng_assert(SingleDestination != PlaceHolderTarget);
      }

      // If we found a `BranchInst` candidate for promotion, we substitute it
      // with an unconditional branch
      if (SingleDestination) {
        IRBuilder<> Builder(Terminator);
        Builder.CreateBr(SingleDestination);

        // We remove the old conditional branch
        Terminator->eraseFromParent();
      }
    }
  } else if (auto *Switch = dyn_cast<SwitchInst>(Terminator)) {

    // Handle the simplification when `PlaceHolderTager` is the default
    // destination of the `SwitchInst`
    BasicBlock *DefaultTarget = Switch->getDefaultDest();
    if (DefaultTarget == PlaceHolderTarget) {

      // We promote the first case, not pointing to `PlaceHolderTarget`. If we
      // promote a case already pointing to `PlaceHolderTarget`, this would, in
      // turn, cause the `default` case to not be simplified ever.
      for (auto CaseIt = Switch->case_begin(); CaseIt != Switch->case_end();
           ++CaseIt) {
        if (CaseIt->getCaseSuccessor() != PlaceHolderTarget) {
          Switch->setDefaultDest(CaseIt->getCaseSuccessor());
          Switch->removeCase(CaseIt);
          break;
        }
      }
    }

    // Handle the simplification when `PlaceHolderTarget` is part the standard
    // `case`s
    for (auto CaseIt = Switch->case_begin(); CaseIt != Switch->case_end();) {
      if (CaseIt->getCaseSuccessor() == PlaceHolderTarget) {

        // We do not want to have a situation where the `PlaceHolderTarget` is
        // both the `default` successor of a `switch` and one of its standard
        // case
        CaseIt = Switch->removeCase(CaseIt);
      } else {
        ++CaseIt;
      }
    }

    // It should never be the case that we end up with a `switch` having only
    // `PlaceHolderTarget` as its successor
    if (Switch->getNumCases() == 0
        and Switch->getDefaultDest() == PlaceHolderTarget) {
      revng_abort();
    }
  }
}

/// Helper function which collects all the nodes reachable from a vector of
/// nodes (`Successors`)
static MapVector<BasicBlock *, SmallSet<BasicBlock *, 4>>
getReachablesFromSuccessor(BasicBlock *N) {

  using ScopeBlock = Scope<BasicBlock *>;
  SetVector<BasicBlock *> Successors;
  for (BasicBlock *Successor : children<ScopeBlock>(N)) {
    Successors.insert(Successor);
  }

  MapVector<BasicBlock *, SmallSet<BasicBlock *, 4>> ReachablesFromSuccessor;
  for (auto *Successor : Successors) {
    SmallSet<BasicBlock *, 4> ReachableExits;

    // Explore all the exits that are reachable from each Successor.
    // For having a divergent exit, we need to find a set of
    // successors that reach only the exit under analysis
    auto &Reachables = ReachablesFromSuccessor[Successor];
    for (auto *DFSNode : depth_first(Scope<BasicBlock *>(Successor))) {
      if (isExit(DFSNode)) {
        Reachables.insert(DFSNode);
      }
    }
  }

  return ReachablesFromSuccessor;
}

/// We define a `Descriptor` for the information needed to identify a
/// `Divergence`. Specifically, the divergence is composed by the conditional
/// where divergence originates, a set of divergent successors, and by the exit
/// node which is divergent wrt. those divergent successors.
struct DivergenceDescriptor {
  BasicBlock *Conditional;
  SmallSet<BasicBlock *, 4> DivergentSuccessors;
  BasicBlock *Exit;
};

/// Helper function that tries to identify a `DivergenceDescriptor`
static std::optional<DivergenceDescriptor>
electDivergence(BasicBlock *Candidate,
                BasicBlock *Exit,
                DomTreeOnView<BasicBlock, Scope> &DomTree) {

  // Check if it is the conditional making it a divergent node
  if (isConditional(Candidate)) {
    revng_assert(DomTree.dominates(Candidate, Exit));

    MapVector<BasicBlock *, SmallSet<BasicBlock *, 4>>
      ReachablesFromSuccessor = getReachablesFromSuccessor(Candidate);
    size_t SuccessorsSize = ReachablesFromSuccessor.size();

    SmallSet<BasicBlock *, 4> DivergentSuccessors;
    for (const auto &[Successor, ReachableExits] : ReachablesFromSuccessor) {
      if (ReachableExits.size() == 1 and *ReachableExits.begin() == Exit) {
        DivergentSuccessors.insert(Successor);
      } else if (ReachableExits.contains(Exit)) {

        // If we reach `Exit` from a successor which is not a candidate for
        // being divergent, it means that `Exit` is reached also by a non
        // divergent successor, and this contradicts the definition, so we
        // cannot find a divergence here, and we return a `nullopt`
        return std::nullopt;
      }
    }

    // We proceed only if there are some divergent exits and some
    // non divergent exits, it doesn't make sense to transform a situation where
    // all the edges are all divergent or all non-divergent
    if (DivergentSuccessors.size() < SuccessorsSize) {

      // We employ this `std::optional` as return value, in order to signal if
      // we identified a candidate for IDS
      return DivergenceDescriptor{ Candidate, DivergentSuccessors, Exit };
    }
  }

  // When no divergence was found, we signal it by returning `nullopt`
  return std::nullopt;
}

/// Helper function that performs the IDS transformation
static void performIDS(DivergenceDescriptor &DivergenceDescriptor,
                       BasicBlock *PlaceHolderTarget) {

  // Create the new `BasicBlock` representing the `C'` conditional
  // inserted by the IDS transformation. We will refer to this block as the
  // `Tail` (of the IDS group of nodes).
  BasicBlock *Conditional = DivergenceDescriptor.Conditional;
  auto DivergentSuccessors = DivergenceDescriptor.DivergentSuccessors;
  BasicBlock *Exit = DivergenceDescriptor.Exit;
  LLVMContext &Context = getContext(Conditional);
  Function *F = Conditional->getParent();
  BasicBlock *Tail = BasicBlock::Create(Context,
                                        Conditional->getName() + "_ids",
                                        F);

  revng_assert(Tail->empty());

  // We clone the terminator already present in `BasicBlock` `Conditional`, so
  // that a superset of the correct final successors are already connected to
  // `Tail`.
  Instruction *ConditionalTerminator = Conditional->getTerminator();
  Instruction *TailTerminator = ConditionalTerminator->clone();
  IRBuilder<> TailBuilder(Tail);
  TailBuilder.Insert(TailTerminator);

  // We connect the `Conditional` to `Tail`, by making sure that all
  // the previous slots and cases going to the nondivergent exits, are now
  // connected to the `Tail` block, in order to preserve the original semantics.
  // The original paths going to the nondivergent exits, are preserved by the
  // the fact that the `Terminator` has been cloned into `Tail`.
  SmallVector<BasicBlock *> NonDivergentSuccessors;
  for (BasicBlock *Successor : children<Scope<BasicBlock *>>(Conditional)) {
    if (not DivergentSuccessors.contains(Successor)) {
      NonDivergentSuccessors.push_back(Successor);
    }
  }
  revng_assert(not DivergentSuccessors.empty()
               and not NonDivergentSuccessors.empty());
  for (BasicBlock *Successor : NonDivergentSuccessors) {
    ConditionalTerminator->replaceSuccessorWith(Successor, Tail);
  }

  // We remove from the `Terminator` of `Tail`, all the edges that
  // target `DivergentSuccessor`, since it will be only reached by
  // `Conditional`. We do this in two steps, we first substitute the original
  // successor with `PlaceHolderTarget`, and then we invoke
  // `simplifyTerminator`, which will take care of simplifying away the
  // unnecessary successors, both for `brcond`s and `switch`es.
  for (BasicBlock *Successor : DivergentSuccessors) {
    TailTerminator->replaceSuccessorWith(Successor, PlaceHolderTarget);
  }
  simplifyTerminator(Tail, PlaceHolderTarget);

  // We add a `scope_closer` edge between the divergent exit node and
  // the `Tail` node
  ScopeGraphBuilder SGBuilder(F);
  SGBuilder.addScopeCloser(Exit, Tail);
}

/// This helper function is used to attempt the IDS process, and returns `true`
/// or `false` depending on whether a change is performed
static bool tryIDS(Function &F, BasicBlock *PlaceHolderTarget) {

  // The main need for recomputing the `DomTree` is that we insert the new
  // `IDS` block and redirect edges over the `ScopeGraph`
  DomTreeOnView<BasicBlock, Scope> DomTree;
  DomTree.recalculate(F);

  // Collect the exit nodes
  SmallVector<BasicBlock *> Exits = getExits(F, PlaceHolderTarget);

  // Attempt IDS on each exit node
  for (BasicBlock *Exit : Exits) {

    // Here we go up in the `ScopeGraph`, hopping through the immediate
    // dominators of the `Exit` block with the goal of finding the divergent
    // conditional
    BasicBlock *Candidate = Exit;
    while ((Candidate = getImmediateDominator(Candidate, DomTree))) {

      // We use the following `std::optional` in order to contain the
      // divergent node candidate, with all the object needed to perform the
      // transformation
      std::optional<DivergenceDescriptor>
        DivergenceDescriptor = electDivergence(Candidate, Exit, DomTree);

      // If we have found a divergence for the exit under analysis, we proceed
      // to perform IDS
      if (DivergenceDescriptor) {
        performIDS(*DivergenceDescriptor, PlaceHolderTarget);

        // Empirically, we have found that restarting the analysis, by
        // recollecting the exit nodes in the `ScopeGraph`, is faster than
        // continuing with the processing of all the exits already collected.
        // Therefore, we should not really try to optimize this, unless we
        // find new evidence that this is better.
        return true;
      }
    }
  }

  // If no change has been performed, we return `false` to signal this fact
  return false;
}

/// Implementation class used to run the `IDS` transformation
class InlineDivergentScopesImpl {
  Function &F;

public:
  InlineDivergentScopesImpl(Function &F) : F(F) {}

public:
  bool run() {

    // We keep a boolean variable to track whether the `Function` was modified
    bool FunctionModified = false;

    // Manager object for the creation and deletion of the `PlaceHolderTarget`
    PlaceHolderTargetManager PlaceHolderTarget(F);

    // Every time we perform a change due to the IDS restructuring, we may have
    // unlocked the potential to perform new IDS closures in nested subtree
    // where the last modification was performed, therefore we need to retry
    // IDS. Empirically, we have found that restarting the analysis, by
    // recollecting the exit nodes in the `ScopeGraph`, is faster than
    // continuing with the processing of all the exits already collected.
    // Therefore, we should not really try to optimize this, unless we find new
    // evidence that this is better.
    while (tryIDS(F, *PlaceHolderTarget)) {

      // As soon as one IDS change is performed, we mark the current `Function`
      // as modified
      FunctionModified = true;
    }

    return FunctionModified;
  }
};

char InlineDivergentScopesPass::ID = 0;
static constexpr const char *Flag = "inline-divergent-scopes";
using Reg = llvm::RegisterPass<InlineDivergentScopesPass>;
static Reg X(Flag,
             "Perform the inline of divergent scopes canonicalization process");

bool InlineDivergentScopesPass::runOnFunction(llvm::Function &F) {

  // Instantiate and call the `Impl` class
  InlineDivergentScopesImpl IDSImpl(F);
  bool FunctionModified = IDSImpl.run();

  // This pass may transform the CFG by assign some blocks to perform the IDS
  // canonicalization and by redirecting edges on the `ScopeGraph`
  return FunctionModified;
}

void InlineDivergentScopesPass::getAnalysisUsage(llvm::AnalysisUsage &AU)
  const {
  // This pass does not preserve the CFG
}
