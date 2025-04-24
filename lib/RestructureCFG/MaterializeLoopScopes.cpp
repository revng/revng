//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/GenericDomTree.h"

#include "revng/RestructureCFG/GenericRegionInfo.h"
#include "revng/RestructureCFG/MaterializeLoopScopes.h"
#include "revng/RestructureCFG/ScopeGraphAlgorithms.h"
#include "revng/RestructureCFG/ScopeGraphGraphTraits.h"
#include "revng/RestructureCFG/ScopeGraphUtils.h"
#include "revng/Support/Assert.h"

using namespace llvm;

// Debug logger
static Logger<> Log("materialize-loop-scopes");

/// Helper function which inserts the `scope_closer` edge representing the
/// `MaterializeLoopScope` operation result
static void addScopeCloser(const ScopeGraphBuilder &SGBuilder,
                           BasicBlock *Head,
                           BasicBlock *UniqueSuccessor) {

  // We split the `Head` block in order to insert a `scope_start`, so that we
  // can insert the `scope_closer` between `scope_start` and the elected
  // `UniqueSuccessor` materializing the `scope` corresponding to the loop body
  Instruction *HeadTerminator = Head->getTerminator();
  BasicBlock *LoopStart = Head->splitBasicBlock(HeadTerminator, "loop_start");

  // Log the `UniqueSuccessor` to which we are inserting the `scope_closer` to
  revng_log(Log,
            "The elected unique successor is block:"
              << UniqueSuccessor->getName() << "\n");

  // We insert the `scope_closer` edge from the `Head` to the `UniqueSuccessor`
  SGBuilder.addScopeCloser(Head, UniqueSuccessor);
}

/// Helper function used to verify that the elected `Head` contains the metadata
/// attached during the `DAGify` pass
static void verifyHeadMD(BasicBlock *Head) {

  // We check that each `Head` block has a metadata attached during the
  // `DAGify` pass
  Instruction *HeadTerminator = Head->getTerminator();
  auto *MD = HeadTerminator->getMetadata("genericregion-head");

  // We must find the metadata by design
  revng_assert(MD);
  auto *Tuple = dyn_cast_or_null<MDTuple>(MD);
  revng_assert(Tuple);
}

/// Helper to obtain the immediate postdominator `BasicBlock`, if present
static BasicBlock *
getImmediatePostDominator(BasicBlock *N,
                          PostDomTreeOnView<BasicBlock, Scope> &PostDomTree) {
  auto *Node = PostDomTree.getNode(N)->getIDom();
  if (Node) {
    return Node->getBlock();
  } else {
    return nullptr;
  }
}

/// Helper function to populate the sets of exiting and successor candidates
static void
collectExitBlocks(GenericRegion<BasicBlock *> *Region,
                  std::set<BasicBlock *> &ExitingBlocks,
                  std::set<BasicBlock *> &UniqueSuccessorCandidates) {
  for (auto *RegionNode : Region->blocks()) {
    SmallSetVector<BasicBlock *, 2>
      Successors = getScopeGraphSuccessors(RegionNode);
    for (BasicBlock *Successor : Successors) {
      if (not Region->containsBlock(Successor)) {

        // We collect the `Successor`s of the `GenericRegion`
        UniqueSuccessorCandidates.insert(Successor);

        // We collect all the blocks from which there is a exiting edge
        // from the `GenericRegion`
        ExitingBlocks.insert(RegionNode);
      }
    }
  }
}

/// Helper function used to create the `Footer` block, which will become the
/// unique successor of the `GenericRegion`, and to divert the exiting edges to
/// such block
static BasicBlock *
enforceFooterSuccessor(GenericRegion<BasicBlock *> *Region,
                       std::set<BasicBlock *> &ExitingBlocks,
                       std::set<BasicBlock *> &UniqueSuccessorCandidates) {

  revng_assert(ExitingBlocks.size() == 1);
  BasicBlock *UniqueExitingBlock = *ExitingBlocks.begin();

  llvm::SmallSet<BasicBlock *, 4> InternalSuccessors;
  SmallSetVector<BasicBlock *, 2>
    Successors = getScopeGraphSuccessors(UniqueExitingBlock);
  for (BasicBlock *Successor : Successors) {
    if (Region->containsBlock(Successor)) {
      InternalSuccessors.insert(Successor);
    }
  }

  auto *ExitingBlockTerminator = UniqueExitingBlock->getTerminator();

  // We clone the original `Terminator` in the `footer` block
  LLVMContext &Context = getContext(UniqueExitingBlock);
  Function *F = UniqueExitingBlock->getParent();
  BasicBlock *Footer = BasicBlock::Create(Context, "footer", F);
  Instruction *FooterTerminator = ExitingBlockTerminator->clone();
  IRBuilder<> FooterBuilder(Footer);
  FooterBuilder.Insert(FooterTerminator);

  // We divert the edges exiting from the `GenericRegion` to the
  // `Footer` block
  replaceSuccessors(ExitingBlockTerminator, UniqueSuccessorCandidates, Footer);

  // In the `Footer`, we need to remove all the edges not going to the
  // `GenericRegion` successors
  for (BasicBlock *InternalSuccessor : InternalSuccessors) {
    simplifyTerminator(Footer, InternalSuccessor);
  }

  return Footer;
}

/// Helper function used to check for the preconditions needed on the
/// predecessors of the candidate immediate post dominator `UniqueSuccessor`.
/// Specifically, we check that all the predecessors are either in the current
/// `GenericRegion` or in the parent one.
static bool
checkPredecessorsCondition(GenericRegion<BasicBlock *> *Region,
                           GenericRegion<BasicBlock *> *ParentRegion,
                           BasicBlock *UniqueSuccessor) {
  // Check that `UniqueSuccessor` belong to `ParentRegion`
  if (not ParentRegion->containsBlock(UniqueSuccessor)) {

    // Log the motivation for the exclusion of the `UniqueSuccessor`
    // candidate
    revng_log(Log,
              "Candidate unique successor " << UniqueSuccessor->getName()
                                            << " is not in the parent region, "
                                               "and needs to be "
                                               "discarded");

    // If the identified `UniqueSuccessor` is not in the parent
    // `GenericRegion`, we signal this fact
    return false;
  }

  // Check that all the predecessors of the candidate `UniqueSuccessor`
  // are either in the `GenericRegion` under analysis or in its parent
  // `GenericRegion`
  SmallSetVector<BasicBlock *, 2>
    Predecessors = getScopeGraphPredecessors(UniqueSuccessor);
  for (BasicBlock *Predecessor : Predecessors) {
    if ((not Region->containsBlock(Predecessor))
        or (not ParentRegion->containsBlock(Predecessor))) {

      // Log the motivation for the exclusion of the `UniqueSuccessor`
      // candidate
      revng_log(Log,
                "Candidate unique successor " << UniqueSuccessor->getName()
                                              << " has not compatible "
                                                 "predecessors, and needs to "
                                                 "be discarded");

      return false;
    }
  }

  return true;
}

/// Helper function used to check for the preconditions needed on the
/// predecessors of the candidate immediate post dominator `UniqueSuccessor`.
/// Specifically, we check that all the predecessors are dominated by the elect
/// entry of the `GenericRegion` (the `Head`).
static bool checkPredecessorsDominance(BasicBlock *Head,
                                       BasicBlock *UniqueSuccessor) {
  DomTreeOnView<BasicBlock, Scope> DomTree;
  Function *F = Head->getParent();
  DomTree.recalculate(*F);

  // Check that all the predecessors are either dominated by the entry
  // node of the loop
  SmallSetVector<BasicBlock *, 2>
    Predecessors = getScopeGraphPredecessors(UniqueSuccessor);
  for (BasicBlock *Predecessor : Predecessors) {
    if (not DomTree.dominates(Head, Predecessor)) {

      // Log the motivation for the exclusion of the `UniqueSuccessor`
      // candidate
      revng_log(Log,
                "Candidate unique successor " << UniqueSuccessor->getName()
                                              << " has not compatible "
                                                 "predecessors, and needs to "
                                                 "be discarded");

      return false;
    }
  }

  return true;
}

static std::optional<BasicBlock *>
tryEnforceUniqueSuccessor(GenericRegion<BasicBlock *> *Region) {

  // Collect the exiting blocks and the `SuccessorCandidates` of the
  // `GenericRegion`
  std::set<BasicBlock *> ExitingBlocks;
  std::set<BasicBlock *> UniqueSuccessorCandidates;
  collectExitBlocks(Region, ExitingBlocks, UniqueSuccessorCandidates);

  std::optional<BasicBlock *> UniqueSuccessor;

  // A. We iterate over all the successors blocks of each node in the
  //    `GenericRegion`. If there is a unique candidate successor, we can
  //    elect such block as `UniqueSuccessor`..

  // If the loop has a single clearly identified successor, we can proceed
  // with the insertion of the `scope_closer` to the `UniqueSuccessor`
  if (UniqueSuccessorCandidates.size() == 1) {
    UniqueSuccessor = *UniqueSuccessorCandidates.begin();

    return UniqueSuccessor;
  }

  // B. If the `GenericRegion` has a single block with exiting edges, we
  //    insert a `footer` block grouping such exiting edges, and we insert
  //    a `scope_closer` to it
  if (ExitingBlocks.size() == 1 and UniqueSuccessorCandidates.size() > 1) {
    BasicBlock *Footer = enforceFooterSuccessor(Region,
                                                ExitingBlocks,
                                                UniqueSuccessorCandidates);

    return Footer;
  }

  // C. If we did not elect the `UniqueSuccessor` in the previous stages,
  //    we try to navigate up in the post dominator tree until we find the
  //    first immediate post dominator block which is outside the current
  //    `GenericRegion.
  revng_assert(not UniqueSuccessor);
  BasicBlock *Head = Region->getHead();
  Function *F = Head->getParent();
  PostDomTreeOnView<BasicBlock, Scope> PostDomTree;
  PostDomTree.recalculate(*F);
  BasicBlock *Candidate = Head;
  while ((Candidate = getImmediatePostDominator(Candidate, PostDomTree))) {
    if (not Region->containsBlock(Candidate)) {

      // Here we have identified the first node outside the
      // `GenericRegion` which postdominates the entry node. This node
      // will be our candidate for becoming the exit node of the
      // `GenericRegion`.
      UniqueSuccessor = Candidate;
      break;
    }
  }

  // We did not find a `UniqueSuccessor`
  if (not UniqueSuccessor) {
    return std::nullopt;
  }

  // We admit the election of a immediate post dominator block as the
  // `UniqueSuccessor` of a `GenericRegion` only if both the following
  // properties hold:
  // 1: The candidate block must be in the direct parent of the
  //    `GenericRegion`.
  // 2: All the incoming edges into the `UniqueSuccessor` block
  //    originate either from blocks in the `GenericRegion` under
  //    analysis (edges exiting the `GenericRegion`) or in its parent.
  auto *ParentRegion = Region->getParent();

  // We may not have a `ParentRegion`, in that case the `GenericRegion` we
  // are analyzing is at the first level in the `RegionTree`, and the
  // following check is not needed
  if (ParentRegion) {

    // If the preconditions are not satisfied, we cannot identify the
    // `UniqueSuccessor`
    if (not checkPredecessorsCondition(Region,
                                       ParentRegion,
                                       *UniqueSuccessor)) {
      return std::nullopt;
    }
  }

  // We now check that all the predecessors of the candidate immediate
  // post dominator block `UniqueSuccessor` are dominated by the `Head`,
  // which is another precondition for the election of the
  // `UniqueSuccessor` of the `GenericRegion`
  if (not checkPredecessorsDominance(Head, *UniqueSuccessor)) {
    return std::nullopt;
  }

  // If we reach this point, it means that all the preconditions are
  // satisfied, and that we can elect the `UniqueSuccessor`
  return UniqueSuccessor;
}

/// Implementation class used to run the `MaterializeLoopScopes`
/// transformation
class MaterializeLoopScopesImpl {
  Function &F;
  ScopeGraphBuilder SGBuilder;

public:
  MaterializeLoopScopesImpl(Function &F) : F(F), SGBuilder(&F) {}

public:
  bool run() {

    // We instantiate and run the `GenericRegionInfo` analysis on the raw CFG,
    // and not on the `ScopeGraph`. This is done because the current pass runs
    // after `DAGify`, which disrupts the loops on the `ScopeGraph`.
    // Under the assumption that the successor order of each `BasicBlock` is not
    // modified between `DAGify` and `MaterializeLoopScopes`, we have the
    // guarantee that the `GenericRegionInfo` computed remains equivalent.
    GenericRegionInfo<Function *> RegionInfo;
    RegionInfo.compute(&F);

    // We keep a boolean variable to track whether the `Function` was modified
    bool FunctionModified = false;

    // We iterate over all the `GenericRegion`s that were found
    for (auto &TopLevelRegion : RegionInfo.top_level_regions()) {
      for (auto *Region : post_order(&TopLevelRegion)) {

        // Retrieve the elected `Head` of the `GenericRegion`
        BasicBlock *Head = Region->getHead();
        revng_log(Log, "Elected head is: " << Head->getName() << "\n");

        // Verify that the `Head` is the same one elected during the `DAGify`
        // pass
        verifyHeadMD(Head);

        // 1. Try to find a `UniqueSuccessor` candidate
        std::optional<BasicBlock *>
          UniqueSuccessor = tryEnforceUniqueSuccessor(Region);

        // 2. Enforce the single successor, if found, by adding a `scope_closer`
        //    to it, for the `GenericRegion`
        if (UniqueSuccessor) {

          // We mark the current `Function` as modified
          FunctionModified = true;

          // We insert the `scope_closer` to `UniqueSuccessor`
          addScopeCloser(SGBuilder, Head, *UniqueSuccessor);
        }
      }
    }

    return FunctionModified;
  }
};

char MaterializeLoopScopes::ID = 0;
static constexpr const char *Flag = "materialize-loop-scopes";
using Reg = llvm::RegisterPass<MaterializeLoopScopes>;
static Reg X(Flag, "Perform the materialization of loop scopes transformation");

bool MaterializeLoopScopes::runOnFunction(llvm::Function &F) {

  // Instantiate and call the `Impl` class
  MaterializeLoopScopesImpl MLoopScopesImpl(F);
  bool FunctionModified = MLoopScopesImpl.run();

  // This pass may transform the CFG by transforming some edges into `goto`
  // edges, and by adding some `scope_closer` edges on the `ScopeGraph`
  return FunctionModified;
}

void MaterializeLoopScopes::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  // This pass does not preserve the CFG
}
