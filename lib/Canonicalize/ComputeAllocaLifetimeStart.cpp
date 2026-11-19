//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Pass.h"

#include "revng/LocalVariables/LifetimeHelpers.h"
#include "revng/LocalVariables/LocalVariableHelpers.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

namespace {

static Logger Log{ "compute-alloca-lifetime-start" };

struct ComputeAllocaLifetimeStartPass : public llvm::FunctionPass {
public:
  static char ID;

  ComputeAllocaLifetimeStartPass() : FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;
};

char ComputeAllocaLifetimeStartPass::ID = 0;

/// Computes and injects a `llvm.lifetime.start` for each alloca that doesn't
/// already have one, before the nearest common dominator of all its uses.
///
/// Very importantly, the alloca is left where it was and the semantics of
/// LLVM IR are preserved. Moving the alloca, which is what we used to do in
/// the past, breaks semantics because it shortens the life of the storage
/// itself, and a value that has to survive a loop iteration is then lost.
///
/// **The assumption** is that every alloca reaching this pass without an
/// already explicit `llvm.lifetime.start` is written before it is read on every
/// path. Every pass in revng that creates allocas currently satisfies this: we
/// allocate in the entry block, store immediately after the instruction whose
/// value the variable stands for, and read it back at that instruction's former
/// uses, so the write dominates every read. `exit-ssa` is the one creator whose
/// variables do not have that shape, since a variable standing for a class of
/// PHIs can be read at the top of a loop and written at the bottom. It marks
/// its own variables, outside any loop they are carried across, and this pass
/// leaves anything already marked alone.
///
/// **If this assumption is broken**, by a pass that creates an alloca read
/// before it is written inside a loop and does not mark it itself, the LLVM IR
/// stays correct but the emitted C does not: the declaration lands in the loop
/// body, so the variable is a fresh, indeterminate one on each iteration, and
/// the value that should have crossed the iteration boundary is lost. Nothing
/// here detects that. A new creator of local variables whose variables can be
/// read before being written has to emit its own marker, the way `exit-ssa`
/// does.
bool ComputeAllocaLifetimeStartPass::runOnFunction(llvm::Function &F) {
  revng_log(Log, "ComputeAllocaLifetimeStart on: " << F.getName());
  LoggerIndent Indent{ Log };

  llvm::DominatorTree DT;
  DT.recalculate(F);

  // Collected before anything is inserted, so that the walk over the function
  // is not disturbed by the markers it emits.
  llvm::SmallVector<llvm::AllocaInst *> Unmarked;

  // `F` is iterated in block order, which is stable, so the markers come out in
  // the same order on every run.
  for (llvm::BasicBlock &BB : F) {
    for (llvm::Instruction &I : BB) {

      // Reading the block holding a use as the place where the use happens is
      // only valid without PHINodes: a PHI uses its incoming value on the
      // incoming edge, not in the block holding the PHI, so the block of the
      // use would not dominate the point where the value is actually needed.
      revng_assert(not llvm::isa<llvm::PHINode>(I),
                   "No PHINodes should reach this point");

      auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(&I);
      if (Alloca == nullptr)
        continue;

      // The stack frame is live for the whole function, and the clifter gives
      // it a declaration of its own.
      if (hasStackFrameMetadata(Alloca))
        continue;

      // If a variable already has a lifetime start, leave it alone.
      if (hasLifetimeStart(Alloca)) {
        revng_log(Log,
                  "skipping " << dumpToString(Alloca)
                              << ", it already has a lifetime.start");
        continue;
      }

      Unmarked.push_back(Alloca);
    }
  }

  bool Changed = false;
  for (llvm::AllocaInst *Alloca : Unmarked) {
    llvm::CallInst *LifetimeStart = emitLifetimeStart(Alloca,
                                                      {},
                                                      DT,
                                                      Alloca->getDebugLoc());

    // An alloca nothing uses has no point at which it becomes live. It is left
    // to the clifter, which declares it where the alloca sits.
    if (LifetimeStart == nullptr)
      continue;

    revng_log(Log,
              "Created lifetime.start: "
                << dumpToString(LifetimeStart)
                << " in Block: " << LifetimeStart->getParent()->getName());
    Changed = true;
  }

  return Changed;
}

using Register = llvm::RegisterPass<ComputeAllocaLifetimeStartPass>;
static Register X("compute-alloca-lifetime-start",
                  "Marks with a llvm.lifetime.start the point "
                  "dominating all the uses of each alloca");

} // namespace
