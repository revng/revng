/// Plan integer rewrites using LLVM value ranges and demanded bits.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/DemandedBits.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"

#include "revng/TypeShrinking/TypeShrinkingAnalysis.h"

#include "SCCPValueRanges.h"

using namespace llvm;

namespace TypeShrinking {

AnalysisKey TypeShrinkingAnalysisPass::Key;

TypeShrinkingAnalysisPass::Result
TypeShrinkingAnalysisPass::run(Function &F, FunctionAnalysisManager &) {
  RewritePlans Result(F);

  // Leave functions with an `invoke` alone. A cast adapting an incoming value
  // of a PHI goes before the terminator of the predecessor, where the result
  // of an `invoke` does not exist yet.
  for (BasicBlock &Block : F) {
    if (isa<InvokeInst>(Block.getTerminator()))
      return Result;
  }

  AssumptionCache AC(F);
  DominatorTree DT(F);
  TargetLibraryInfoImpl Impl(Triple(F.getParent()->getTargetTriple()));
  TargetLibraryInfo TLI(Impl);

  // Solve ranges with SCCP, including loop invariants. Use them to refine
  // backward demand before creating rewrite plans.
  SCCPValueRanges Ranges(F, TLI);
  DemandedBits Demands(F, AC, DT, [&Ranges](const Use &U) {
    return Ranges.range(U.get());
  });

  // Retain a low-bit prefix covering LLVM's demanded mask. Range queries
  // finish before rebuilding invalidates either analysis.
  ReversePostOrderTraversal<Function *> Blocks(&F);
  for (BasicBlock *Block : Blocks) {
    for (Instruction &I : *Block) {
      if (not I.getType()->isIntegerTy() or Demands.isInstructionDead(&I))
        continue;

      unsigned Demand = Demands.getDemandedBits(&I).getActiveBits();

      // This pass leaves dead code for DCE, but a dead instruction still
      // executes, and demand analysis attributes nothing to it. Narrowing an
      // operand it reads could therefore change what it computes:
      //
      //     %d = or i64 %x, 256       ; never zero, bit 8 is always set
      //     %unused = udiv i64 42, %d ; dead, but defined for every %x
      //     %r = trunc i64 %d to i8   ; the only live use wants 8 bits
      //
      // Narrowing %d to the 8 demanded bits drops bit 8 and leaves %unused
      // dividing by zero. Count a dead user as reading every bit, which keeps
      // such an operand wide. Narrowing driven by \ref SCCPValueRanges stays
      // available, as bounds hold for the value a dead user reads too.
      for (User *U : I.users()) {
        if (Demands.isInstructionDead(cast<Instruction>(U))) {
          Demand = I.getType()->getIntegerBitWidth();
          break;
        }
      }

      if (auto Plan = RewritePlan::create(I, Demand, Ranges))
        Result.insert(I, Plan);
    }
  }
  return Result;
}

} // namespace TypeShrinking
