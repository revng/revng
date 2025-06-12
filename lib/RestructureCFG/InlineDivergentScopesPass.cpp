//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/GenericDomTree.h"

#include "revng/ADT/Concepts.h"
#include "revng/RestructureCFG/InlineDivergentScopesPass.h"
#include "revng/RestructureCFG/ScopeGraphAlgorithms.h"
#include "revng/RestructureCFG/ScopeGraphGraphTraits.h"
#include "revng/RestructureCFG/ScopeGraphUtils.h"
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

/// Helper function which detects if `Node` is an exit for the `ScopeGraph`
static bool isExit(BasicBlock *Node) {
  auto Successors = children<Scope<BasicBlock *>>(Node);
  return std::ranges::empty(Successors);
}

/// Helper function to retrieve the target BasicBlock of a `GotoBlock`
static BasicBlock *getGotoTarget(const BasicBlock *BB) {
  revng_assert(isGotoBlock(BB));
  const Instruction *Terminator = BB->getTerminator();
  const BranchInst *Branch = llvm::cast<BranchInst>(Terminator);
  revng_assert(Branch->isUnconditional());
  return Branch->getSuccessor(0);
}

/// Helper function to collect all the exit nodes on the `ScopeGraph`
static SmallVector<BasicBlock *> getExits(Function &F,
                                          BasicBlock *PlaceHolderTarget) {
  SmallVector<BasicBlock *> TrueExits;
  SmallVector<BasicBlock *> GotoExits;
  Scope<Function *> ScopeGraph(&F);
  for (BasicBlock *BB : nodes(ScopeGraph)) {

    // We need to exclude the `PlaceHolderTarget` which is a temporary block
    // which incidentally happens to be an exit, but will be removed before the
    // end of the pass
    if (BB != PlaceHolderTarget and isExit(BB)) {
      if (not isGotoBlock(BB)) {
        TrueExits.push_back(BB);
      } else {
        GotoExits.push_back(BB);
      }
    }
  }

  // We now order the `GotoExits` by placing first the ones whose `goto` target
  // comes first in `post_order` in the graph. This criterion should maximize
  // the opportunities of performing `MaterializeTrivialGoto` transformations.

  // Create a map to store the `post_order` indexes, that we can later reuse
  // when performing the comparison
  llvm::DenseMap<const BasicBlock *, unsigned> POIndexes;

  // Fill the map
  unsigned Index = 0;
  for (BasicBlock *BB : post_order(ScopeGraph)) {
    POIndexes[BB] = Index;
    Index++;
  }

  // Sort the `GotoExits` according to the above criterion
  std::sort(GotoExits.begin(),
            GotoExits.end(),
            [&POIndexes](const BasicBlock *A, const BasicBlock *B) {
              return POIndexes[getGotoTarget(A)] < POIndexes[getGotoTarget(B)];
            });

  // Merge the `TrueExits` with the `GotoExits`
  GotoExits.append(TrueExits);
  return GotoExits;
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

/// Define a concept to restrict the usage of `replaceSuccessors` over a
/// `SuccessorsToRemove` parameter of type the `SmallSet`s and `SmallVector`
template<typename T, typename ValueType>
concept IterableOfBasicBlockPtrs = requires(T Container) {
  { std::begin(Container) } -> std::input_iterator;
  { std::end(Container) };
  { *std::begin(Container) } -> std::convertible_to<ValueType>;
};

/// Helper function which substitutes some successors in the `Terminator` with
/// `NewTarget`
template<IterableOfBasicBlockPtrs<BasicBlock *> Container>
static void replaceSuccessors(Instruction *Terminator,
                              Container &SuccessorsToRemove,
                              BasicBlock *NewTarget) {
  for (BasicBlock *Successor : SuccessorsToRemove) {
    Terminator->replaceSuccessorWith(Successor, NewTarget);
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

/// Helper function which create a single entry point block where a divergence
/// is entered into by multiple successors of a `Conditional`
static void createHead(BasicBlock *Conditional,
                       DivergenceDescriptor &Divergence,
                       BasicBlock *PlaceHolderTarget) {
  LLVMContext &Context = getContext(Conditional);
  Function *F = Conditional->getParent();
  Instruction *ConditionalTerminator = Conditional->getTerminator();
  BasicBlock *Head = BasicBlock::Create(Context,
                                        Conditional->getName() + "_head_ids",
                                        F);
  Instruction *HeadTerminator = ConditionalTerminator->clone();
  IRBuilder<> HeadBuilder(Head);
  HeadBuilder.Insert(HeadTerminator);

  // Collect the `Successor`s composing the local `Divergence`
  SmallSet<BasicBlock *, 4> LocalDivergentSuccessors = Divergence
                                                         .DivergentSuccessors;

  // And collect all the `Successor`s that do not make up the local
  // `Divergence`
  SmallVector<BasicBlock *> LocalNonDivergentSuccessors;
  for (BasicBlock *Successor : children<Scope<BasicBlock *>>(Head)) {
    if (not LocalDivergentSuccessors.contains(Successor)) {
      LocalNonDivergentSuccessors.push_back(Successor);
    }
  }

  // The `Head` only connects the `LocalDivergentSuccessors`, we simplify
  // away the other successors
  replaceSuccessors(HeadTerminator,
                    LocalNonDivergentSuccessors,
                    PlaceHolderTarget);
  simplifyTerminator(Head, PlaceHolderTarget);

  // If we insert the `Head`, we replace all the edges going to the
  // `DivergentSuccessors` in the conditional so that they go to `Head`
  replaceSuccessors(ConditionalTerminator, LocalDivergentSuccessors, Head);
}

/// Helper function that performs the IDS transformation
static void
performMultipleIDS(const ScopeGraphBuilder &SGBuilder,
                   SmallVector<DivergenceDescriptor> &MultipleDivergences,
                   BasicBlock *PlaceHolderTarget) {

  // Create the new `BasicBlock` representing the `C'` conditional
  // inserted by the IDS transformation. We will refer to this block as the
  // `Tail` (of the IDS group of nodes).
  BasicBlock *Conditional = MultipleDivergences[0].Conditional;
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

  // Populate the `DivergentSuccessors` summing all ones present in each
  // `DivergenceDescriptor`
  SmallSet<BasicBlock *, 4> DivergentSuccessors;
  for (auto Divergence : MultipleDivergences) {
    DivergentSuccessors.insert(Divergence.DivergentSuccessors.begin(),
                               Divergence.DivergentSuccessors.end());
  }

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

  revng_assert(not DivergentSuccessors.empty());

  // It may be that for a specific `Conditional` we elected all the exits as
  // divergent. In this case, we need to arbitrarily elect one exit as the `one
  // true exit`, and we do this by electing the last element in
  // `MultipleDivergences`.
  if (NonDivergentSuccessors.empty()) {
    revng_assert(MultipleDivergences.size() > 1);

    // The fact that we elect the last element in `MultipleDivergences` as the
    // only non divergent exit, is very important, due to how the `exit` blocks
    // are enqueued in `MupltipleDivergences`. If present, the `Goto` exit will
    // be in the first positions, followed by the original exit blocks. In this
    // way, if present, we always elect a original exit as the non divergent
    // exit.
    size_t LastElementIndex = MultipleDivergences.size() - 1;

    // The first element in `MultipleDivergencies` is arbitrarily elected as the
    // component not going to a divergent exit
    NonDivergentSuccessors
      .append(MultipleDivergences[LastElementIndex].DivergentSuccessors.begin(),
              MultipleDivergences[LastElementIndex].DivergentSuccessors.end());

    // We remove the `Successor`s from the `DivergentSuccessors`
    for (BasicBlock *Successor :
         MultipleDivergences[LastElementIndex].DivergentSuccessors) {
      DivergentSuccessors.erase(Successor);
    }

    // We erase the first `DivergenceDescriptor` from the `MultipleDivergences`
    MultipleDivergences.erase(MultipleDivergences.begin() + LastElementIndex);
  }

  // Decide if we need a new `Head` that collects the entry of the divergent
  // scope. This is needed if one `Divergence` is composed by multiple
  // `Successor`s, since in that case there is no clear entry to the inlined
  // divergent scope. We therefore add a `Head` block that collects the entry
  // point, so that this new node will dominate the eventual postdominator
  // common to the multiple `Successor`s.
  for (auto Divergence : MultipleDivergences) {
    if (Divergence.DivergentSuccessors.size() > 1) {
      createHead(Conditional, Divergence, PlaceHolderTarget);
    }
  }

  // Move the `NonDivergentSuccessors` outgoing edges from `Conditional` so that
  // they point to `Tail`
  replaceSuccessors(ConditionalTerminator, NonDivergentSuccessors, Tail);

  // We remove from the `Terminator` of `Tail`, all the edges that
  // target `DivergentSuccessor`, since it will be only reached by
  // `Conditional`. We do this in two steps, we first substitute the original
  // successor with `PlaceHolderTarget`, and then we invoke
  // `simplifyTerminator`, which will take care of simplifying away the
  // unnecessary successors, both for `brcond`s and `switch`es.
  replaceSuccessors(TailTerminator, DivergentSuccessors, PlaceHolderTarget);
  simplifyTerminator(Tail, PlaceHolderTarget);

  // We add a `scope_closer` edge between the divergent exit node and
  // the `Tail` node for all the exits
  for (auto Divergence : MultipleDivergences) {
    SGBuilder.addScopeCloser(Divergence.Exit, Tail);
  }
}

/// Helper function which returns if a `DivergenceDescriptor` is compatible with
/// all the ones already present in a `Collection`. Being compatible means that
/// they insist on the same conditional, have non-overlapping `Exit` blocks, and
/// do not have any overlapping `DivergentSuccessors`. We currently use this
/// helper only in an assertion to double check that the `Divergencies`
/// collected respect the compatibility criterion.
static bool isCompatible(const SmallVector<DivergenceDescriptor> &Collection,
                         const DivergenceDescriptor &NewDescriptor) {

  // A new element is always compatible with an empty `Collection`
  if (Collection.empty())
    return true;

  // A new element is compatible when it shares the `Conditional` with the ones
  // already in the `Collection`
  if (Collection[0].Conditional != NewDescriptor.Conditional)
    return false;

  // All the `DivergenceDescriptor`s involving a certain `Conditional`, must
  // partition, so that there is no overlap: 1: the divergent `Exit` reached by
  // each `DivergenceDescriptor` 2: the set of the `DivergentSuccessors`
  for (const auto &Existing : Collection) {
    if (Existing.Exit == NewDescriptor.Exit) {
      return false;
    }

    for (BasicBlock *ExistingSucc : Existing.DivergentSuccessors) {
      for (BasicBlock *NewSucc : NewDescriptor.DivergentSuccessors) {
        if (ExistingSucc == NewSucc) {
          return false;
        }
      }
    }
  }

  return true;
}

/// Helper function which attempts to perform multiple IDS transformations
static void
tryMultipleIDS(const ScopeGraphBuilder &SGBuilder,
               const SmallVector<DivergenceDescriptor> &DivergenceDescriptors,
               BasicBlock *PlaceHolderTarget) {

  // We attempt to perform multiple IDS transformations all at once
  // We start by collecting all the `Conditional`s involved in at least one
  // divergence
  SmallSetVector<BasicBlock *, 4> Conditionals;
  for (auto DivergenceDescriptor : DivergenceDescriptors) {
    Conditionals.insert(DivergenceDescriptor.Conditional);
  }

  // For each `Conditional`, we collect all the multiple divergences involving
  // it
  for (BasicBlock *Conditional : Conditionals) {

    // Support variables used to understand when we need to restart the
    // collection
    // of the `DivergentDescriptor`s
    bool AllExitGotos = true;
    size_t CoveredSuccessors = 0;
    size_t ConditionalSuccessors = getScopeGraphSuccessors(Conditional).size();

    SmallVector<DivergenceDescriptor> CompatibleDivergenceDescriptors;
    for (auto DivergenceDescriptor : DivergenceDescriptors) {
      if (DivergenceDescriptor.Conditional == Conditional) {

        // As a safeguard, we assert that all the `DivergenceDescriptor`s are
        // compatible, i.e., they partition correctly the divergencies insisting
        // on a certain `Conditional`
        revng_assert(isCompatible(CompatibleDivergenceDescriptors,
                                  DivergenceDescriptor));
        CompatibleDivergenceDescriptors.push_back(DivergenceDescriptor);

        // Update the `AllExitGotos` state with the information of the current
        // `Exit`
        AllExitGotos = AllExitGotos and isGotoBlock(DivergenceDescriptor.Exit);

        // Update the `CoveredSuccessors` information with the current
        // divergence
        CoveredSuccessors += DivergenceDescriptor.DivergentSuccessors.size();
      }
    }

    // We must find at least one `DivergenceDescriptor` for each `Conditional`,
    // since we previously selected only `Conditional`s involved in at least
    // one `Divergence`
    revng_assert(CompatibleDivergenceDescriptors.size() >= 1);

    performMultipleIDS(SGBuilder,
                       CompatibleDivergenceDescriptors,
                       PlaceHolderTarget);

    // Criterion which restarts the collection and processing of the divergent
    // exits before every already collected divergence is processed.
    // This must happen when:
    // 1) The `Conditional` has only divergent successors.
    // 2) All the divergent successors lead to a `goto` exit.
    // The reasoning is the following: if a `Conditional` has all `goto`
    // divergent successors, once we perform IDS for such `Conditional` one of
    // the successors will be elected as the non divergent one, and the
    // corresponding exit node, which is a `goto` exit, may become a new
    // divergent exit for another `Conditional` upper in the `ScopeGraph`. In
    // such case, if we do not restart the collection of the divergent exits, we
    // may process divergences already collected, before the newly introduced
    // one. And this violates the invariant that we always process divergent
    // `goto` exits before the non `goto` exits. And this in turn may cause
    // suboptimalities when `IDS` is combined with `MaterializeTrivialGoto` in
    // order to reduce the number of emitted `goto`s.
    if (AllExitGotos and CoveredSuccessors == ConditionalSuccessors)
      return;
  }
}

/// This helper function is used to attempt the IDS process, and returns `true`
/// or `false` depending on whether a change is performed
static bool tryIDS(const ScopeGraphBuilder &SGBuilder,
                   Function &F,
                   BasicBlock *PlaceHolderTarget) {

  // The main need for recomputing the `DomTree` is that we insert the new
  // `IDS` block and redirect edges over the `ScopeGraph`
  DomTreeOnView<BasicBlock, Scope> DomTree;
  DomTree.recalculate(F);

  SmallVector<DivergenceDescriptor> DivergenceDescriptors;

  // Collect the exit nodes
  SmallVector<BasicBlock *> Exits = getExits(F, PlaceHolderTarget);

  // Collect all the IDS opportunities.
  // We iterate over all the `Exit`s, and then search for a `Candidate`
  // conditional node for which such `Exit` is divergent wrt. We do this by
  // walking upward the dominator tree until we find (if it is present) the
  // divergent `Conditional`. An alternative, which would also make the code
  // more straightforward, would be to iterate over the `Conditional`s and start
  // the search from there, but this would mean to perform multiple visits to
  // the `Exit`s. With this techniques instead, even if less intuitive, we
  // collect all the `DivergenceDescriptor`s in one sweep.
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

      // If we have found a divergence for the exit under analysis, we add it to
      // the `DivergenceDescriptors`
      if (DivergenceDescriptor) {
        DivergenceDescriptors.push_back(*DivergenceDescriptor);

        // We have found a `Divergence` for the current `Exit`, so we can move
        // to the next one
        break;
      }
    }
  }

  // If we did not collect any `DivergenceDescriptor`, we return `false` in
  // order to signal that no change was performed at the last round
  if (DivergenceDescriptors.empty()) {
    return false;
  } else {
    tryMultipleIDS(SGBuilder, DivergenceDescriptors, PlaceHolderTarget);
    return true;
  }
}

/// Implementation class used to run the `IDS` transformation
class InlineDivergentScopesImpl {
  Function &F;
  const ScopeGraphBuilder SGBuilder;

public:
  InlineDivergentScopesImpl(Function &F) : F(F), SGBuilder(&F) {}

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
    while (tryIDS(SGBuilder, F, *PlaceHolderTarget)) {

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
