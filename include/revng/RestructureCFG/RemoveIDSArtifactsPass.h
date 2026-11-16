#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

/// Undo the `switch` splitting that `inline-divergent-scopes` leaves behind.
///
/// IDS inlines a divergent scope by cloning the terminator of the divergent
/// conditional into a fresh block, the `Tail`, redirecting the non divergent
/// edges to it and adding a `scope_closer` from each divergent exit back to
/// it. When the conditional is a `switch`, the `Tail` ends up holding a
/// `switch` on the very same condition, dispatching the values the conditional
/// hands over to it. Repeated IDS rounds stack these, so one `switch` ends up
/// spread over several blocks:
///
///     cond:   switch i32 %v, label %t0 [ i32 0, label %a, i32 1, label %t0 ]
///     t0:     switch i32 %v, label %t1 [ i32 0, label %t1, i32 1, label %b ]
///     t1:     switch i32 %v, label %c  [ i32 0, label %c,  i32 1, label %c  ]
///
/// The pass never reasons about that as a chain. It visits the blocks in post
/// order on the `ScopeGraph` and folds one block at a time into the single
/// block that can branch to it, when the two `switch` on the same value. A
/// chain collapses because post order visits the last link first: once it has
/// been folded into the link before it, that link is itself a foldable block by
/// the time it is visited.
///
/// The fold is local. Both blocks `switch` on the same value and the folded
/// block is reached from one block alone, so on every edge between them the
/// value is pinned down by the edge itself: an edge labelled `v` ends up
/// wherever the folded block sends `v`, and the default edge, which carries
/// everything the predecessor does not label, keeps being dispatched by the
/// folded block's own cases and default.
///
/// Expects the `ScopeGraph` to have a single exit, which `enforce-single-exit`
/// guarantees where this runs in the pipeline, and leaves it that way. The
/// `scope_closer`s aimed at a folded block are retargeted rather than cut for
/// that reason: a `goto` block exposes its `scope_closer` target as its only
/// `ScopeGraph` successor, so dropping the marker would turn the block into a
/// second exit, cost the conditionals above it their postdominator, and make
/// the clifter emit whatever the paths below them reconverge on once per path.
///
/// Expects the function to hold no `phi`, which `exit-ssa` guarantees well
/// before IDS runs, and asserts it.
///
/// Returns true when `F` was modified.
bool removeIDSArtifacts(llvm::Function &F);

class RemoveIDSArtifactsPass
  : public llvm::PassInfoMixin<RemoveIDSArtifactsPass> {
public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM) {
    if (not removeIDSArtifacts(F))
      return llvm::PreservedAnalyses::all();

    return llvm::PreservedAnalyses::none();
  }
};

/// `revng opt -remove-ids-artifacts`, and the `pure-llvm-passes` pipes that
/// name it in `pipeline.yml`, still go through the legacy pass manager.
class RemoveIDSArtifactsLegacyPass : public llvm::FunctionPass {
public:
  static char ID;

public:
  RemoveIDSArtifactsLegacyPass() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override {
    return removeIDSArtifacts(F);
  }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {}
};
