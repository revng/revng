//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/Support/GenericDomTree.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "revng/ADT/ReversePostOrderTraversal.h"
#include "revng/RestructureCFG/MaterializeTrivialGotoPass.h"
#include "revng/RestructureCFG/ScopeGraphAlgorithms.h"
#include "revng/RestructureCFG/ScopeGraphGraphTraits.h"
#include "revng/RestructureCFG/ScopeGraphUtils.h"
#include "revng/Support/Debug.h"
#include "revng/Support/GraphAlgorithms.h"

using namespace llvm;

// Debug logger
Logger MaterializeTrivialGotoLogger("materialize-trivial-goto");

static void eraseGoto(ScopeGraphBuilder &SGBuilder, BasicBlock &BB) {

  // If `BB` is a `GotoBlock`, it must have as terminator an unconditional
  // branch, which points to the `goto` target block.
  Instruction *GotoTerminator = BB.getTerminator();
  BranchInst *Branch = cast<BranchInst>(GotoTerminator);
  revng_assert(Branch->isUnconditional());

  // We erase the `goto_block` marker
  SGBuilder.eraseGoto(&BB);
}

static BasicBlock *eraseScopeCloser(ScopeGraphBuilder &SGBuilder,
                                    BasicBlock &BB) {

  // If the `GotoBlock` also contains a `scope_closer` edge, we also need
  // to remove it, and to restore it in case we need to rollback the
  // transformation. We return it so that it can be later restored
  BasicBlock *ScopeCloserTarget = nullptr;
  if (isScopeCloserBlock(&BB)) {
    ScopeCloserTarget = SGBuilder.eraseScopeCloser(&BB);
  }
  return ScopeCloserTarget;
}

static void rollbackScopeGraph(ScopeGraphBuilder &SGBuilder,
                               BasicBlock &BB,
                               BasicBlock *ScopeCloserTarget) {
  SGBuilder.makeGoto(&BB);

  // If there was also a `scope_closer` in addition to the `GotoBlock`, we need
  // to restore it too
  if (ScopeCloserTarget) {
    SGBuilder.addScopeCloser(&BB, ScopeCloserTarget);
  }
}

class MaterializeTrivialGotoPassImpl {
  Function &F;
  ScopeGraphBuilder SGBuilder;

public:
  MaterializeTrivialGotoPassImpl(Function &F) : F(F), SGBuilder(&F) {}

public:
  bool run() {
    bool FunctionModified = false;

    // We store a `SmallVector` of the `GotoBlocks` whose `goto`s are removed,
    // so that we can simplify them all away at the end
    llvm::SmallVector<BasicBlock *> SimplifiedGotoBlocks;

    // We iterate over all the blocks in the function, searching for `goto`
    // blocks
    for (BasicBlock &BB : F) {
      if (isGotoBlock(&BB)) {

        // We remove the `goto_block` marker (and the `scope_closer` if
        // present)
        eraseGoto(SGBuilder, BB);
        BasicBlock *ScopeCloserTarget = eraseScopeCloser(SGBuilder, BB);

        // We rollback the changes if either:
        // 1) The obtained `ScopeGraph` becomes cyclic
        // 2) The obtained `ScopeGraph` becomes undecided
        if (not isDAG<Scope<Function *>, Scope<BasicBlock *>>(&F)
            or not isScopeGraphDecided(F)) {

          // We rollback to the original situation
          rollbackScopeGraph(SGBuilder, BB, ScopeCloserTarget);
        } else {

          // If we do not rollback, it means that the `TrivialGoto`
          // `Materialization` operation stuck, and therefore the `LLVMIR` has
          // been modified
          FunctionModified = true;

          // The `GotoBlock` is at this point useless, we save it for possibly
          // simplifying them later
          SimplifiedGotoBlocks.push_back(&BB);
        }
      }
    }

    // We try to batch remove all the `GotoBlock`s which are not `goto` anymore
    for (BasicBlock *GotoBlock : SimplifiedGotoBlocks) {
      MergeBlockIntoPredecessor(GotoBlock);
    }

    // We verify that the `ScopeGraph` has not blocks disconnected from the
    // entry block
    if (VerifyLog.isEnabled()) {
      revng_assert(not hasUnreachableBlocks(&F));
    }

    return FunctionModified;
  }
};

char MaterializeTrivialGotoPass::ID = 0;
static constexpr const char *Flag = "materialize-trivial-goto";
using Reg = llvm::RegisterPass<MaterializeTrivialGotoPass>;
static Reg X(Flag, "Perform the MaterializeTrivialGoto pass on the ScopeGraph");

bool MaterializeTrivialGotoPass::runOnFunction(llvm::Function &F) {

  // Instantiate and call the `Impl` class
  MaterializeTrivialGotoPassImpl MaterializeTrivialGotoImpl(F);
  bool FunctionModified = MaterializeTrivialGotoImpl.run();

  // This pass may transform the CFG by transforming some `goto` edges into
  // standard edges. We propagate the information computed by the `Impl` class.
  return FunctionModified;
}

void MaterializeTrivialGotoPass::getAnalysisUsage(llvm::AnalysisUsage &AU)
  const {
  // This pass does not preserve the CFG
}
