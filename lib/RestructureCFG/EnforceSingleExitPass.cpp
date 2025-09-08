//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"

#include "revng/RestructureCFG/EnforceSingleExitPass.h"
#include "revng/RestructureCFG/ScopeGraphGraphTraits.h"
#include "revng/RestructureCFG/ScopeGraphUtils.h"
#include "revng/Support/GraphAlgorithms.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

/// Helper function to detect successors of a block on the `ScopeGraph`
static bool hasScopeGraphSuccessors(BasicBlock *BB) {
  return !children<Scope<BasicBlock *>>(BB).empty();
}

/// Helper function to detect an infinite loop on the `ScopeGraph`
static bool isInfiniteLoop(const scc_iterator<Scope<BasicBlock *>> &SCCIt) {

  // We transpile the nodes composing the SCC in a set to have faster lookup
  llvm::SmallSet<BasicBlock *, 4> SCCNodes;
  for (auto *BB : *SCCIt) {
    SCCNodes.insert(BB);
  }

  // We search for any block in the SCC that have an exiting edge (on the
  // `ScopeGraph` too, mind the `GT` parameter)
  for (auto *BB : *SCCIt) {
    using GT = llvm::GraphTraits<Scope<llvm::BasicBlock *>>;
    auto Successors = make_range(GT::child_begin(BB), GT::child_end(BB));
    for (auto Successor : Successors) {
      if (!SCCNodes.contains(Successor)) {
        return false;
      }
    }
  }

  return true;
}

/// Implementation class used to run the Enforce Single Exit transformation.
/// This pass, in presence of a `ScopeGraph` with multiple exit blocks, creates
/// a new `sink_block` where to redirect (on the `ScopeGraph`, with
/// `scope_closer` edges) each exit block, so that the `ScopeGraph` after the
/// transformations has a single `sink_block`. We took inspiration from the
/// internals of the `llvm/include/llvm/Support/GenericDomTreeConstruction.h`
/// header, where a similar operation is done in order to compute the
/// `PostDominatorTree` for a generic CFG, which can contain multiple exit
/// blocks, and infinite loops too.
class EnforceSingleExitPassImpl {
  Function &F;
  const ScopeGraphBuilder SGBuilder;

public:
  EnforceSingleExitPassImpl(Function &F) : F(F), SGBuilder(&F) {}

public:
  bool run() {
    llvm::SmallVector<BasicBlock *> TrivialExits;
    llvm::SmallVector<BasicBlock *> NonTrivialExits;

    // 1: Find all the trivial exits of the graph, i.e., the blocks which do not
    //    have any successors on the `ScopeGraph`
    for (BasicBlock &BB : F) {
      if (not hasScopeGraphSuccessors(&BB)) {
        TrivialExits.push_back(&BB);
      }
    }

    // 2: Find all the non trivial exits, i.e., those which are in infinite
    //    loops. To do this, we iterate over all the SCCs present in the
    //    `ScopeGraph` (mind the `GraphT` type passed to the `scc_iterator`),
    //    and search for components that do not have any exiting edges.
    using GraphT = Scope<BasicBlock *>;
    BasicBlock *Entry = &F.getEntryBlock();
    for (scc_iterator<GraphT> I = scc_begin(Scope(Entry)),
                              IE = scc_end(Scope(Entry));
         I != IE;
         ++I) {

      // We skip any SCC which doesn't compose a cycle
      if (not I.hasCycle()) {
        continue;
      }

      // Detect if the SCC is an infinite loop
      bool IsInfiniteLoop = isInfiniteLoop(I);

      // If we found an SCC having an exiting edge on the `ScopeGraph`, we move
      // on to the next SCC
      if (not IsInfiniteLoop)
        continue;

      // We elect the "furthest away" node, along some DFS, for the elected SCC.
      // Due to the internal behavior of the `scc_iterator`, we can elect the
      // first node in each `SCC` as the furthest away node on the DFS used
      // during the `scc_iterator` itself. This would
      BasicBlock *LastSCCNode = *I->begin();
      NonTrivialExits.push_back(LastSCCNode);
    }

    // We should have collected at least one exit
    revng_assert(TrivialExits.size() + NonTrivialExits.size() != 0);

    // 3: The only situation where we end up not doing any transformation to the
    //    `Function`, is when we have a single `TrivialExit`, and no
    //    `NonTrivialExit`
    if (TrivialExits.size() == 1 and NonTrivialExits.size() == 0)
      return false;

    // 4: If we found at least one `TrivialExit`, we elect that one as the
    //    future `OneTrueExit`. If only `NonTrivialExits` are found, we need to
    //    create an ad-hoc `SinkBlock` which will become the `OneTrueExit`.
    BasicBlock *OneTrueExit;
    if (TrivialExits.size() == 0) {

      // 5: Create the new sink block
      LLVMContext &Context = getContext(&F);
      BasicBlock *SinkBlock = BasicBlock::Create(Context, "sink_block", &F);

      // Add an `UnreachableInst` to the end of the `SinkBlock`
      revng::IRBuilder Builder(Context);
      Builder.SetInsertPoint(SinkBlock);
      Builder.CreateUnreachable();

      // 6: We need to create a new entry block, ending with a conditional
      //    branch (using the `true` constant as condition), whose `true` branch
      //    goes to the original entry `BasicBlock`, and whose `false` branch
      //    goes to newly create sink node. This is needed in order to keep the
      //    `sink_block` alive on the CFG too, because the other edges incoming
      //    are visible on the `ScopeGraph` only.
      BasicBlock *OriginalEntry = &F.getEntryBlock();
      BasicBlock *NewEntryBlock = BasicBlock::Create(Context,
                                                     "new_entry_block",
                                                     &F,
                                                     OriginalEntry);
      Builder.SetInsertPoint(NewEntryBlock);
      Value *ConditionTrue = Builder.getTrue();
      Builder.CreateCondBr(ConditionTrue, OriginalEntry, SinkBlock);

      OneTrueExit = SinkBlock;

    } else {

      // If there is at least one `TrivialExit`, we use the first one as the
      // `OneTrueExit`
      OneTrueExit = *TrivialExits.begin();
    }

    // 7: Connect all the `TrivialExits`, except the first one, to the
    // `OneTrueExit`
    if (TrivialExits.size() != 0) {
      for (BasicBlock *TrivialExit : skip_front(TrivialExits)) {
        revng_assert(TrivialExit != OneTrueExit);
        SGBuilder.addScopeCloser(TrivialExit, OneTrueExit);
      }
    }

    // 8: Connect all the `NonTrivialExits` to the `OneTrueExit`
    for (BasicBlock *NonTrivialExit : NonTrivialExits) {

      // Build a `scope_closer` edge from each identified exit block to the
      // `sink_block`
      SGBuilder.addScopeCloser(NonTrivialExit, OneTrueExit);
    }

    // The function was modified
    return true;
  }
};

char EnforceSingleExitPass::ID = 0;

static constexpr const char *Flag = "enforce-single-exit";
using Reg = llvm::RegisterPass<EnforceSingleExitPass>;
static Reg X(Flag, "Enforce the Single Exit Property on the ScopeGraph");

bool EnforceSingleExitPass::runOnFunction(llvm::Function &F) {

  // Instantiate and call the `Impl` class
  EnforceSingleExitPassImpl ESEImpl(F);
  bool FunctionChanged = ESEImpl.run();

  // We verify that the `ScopeGraph` has not blocks disconnected from the
  // entry block
  if (VerifyLog.isEnabled()) {
    revng_assert(not hasUnreachableBlocks(&F));
  }

  // This pass may transform the CFG by inserting new block, edges, and
  // `scope_closer` edges (which however do not affect the CFG). We propagate
  // this information up to the `FunctionPassManager`
  return FunctionChanged;
}

void EnforceSingleExitPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  // This pass does not preserve the CFG
}
