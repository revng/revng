//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Analysis/IVUsers.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/Transforms/Utils/SimplifyIndVar.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

static Logger<> Log{ "loop-rewrite-with-canonical-induction-variable" };

class LoopRewriteIV {
public:
  LoopRewriteIV(LoopInfo &LI) : LI(LI) {}

  // This transforms the loop's iv in the canonical form.
  bool run(Loop &L, IVUsers *IU, ScalarEvolution *SE, DominatorTree *DT);

private:
  bool isIntWithPowerOfTwoByteSize(const SCEV *S) const;

private:
  LoopInfo &LI;
};

// Return true when S contains a non-power-of-two sized integer.
bool LoopRewriteIV::isIntWithPowerOfTwoByteSize(const SCEV *S) const {
  return SCEVExprContains(S, [](const SCEV *S) {
    if (auto *IntTy = dyn_cast<llvm::IntegerType>(S->getType())) {
      auto IntSize = IntTy->getIntegerBitWidth();
      return IntSize != 1 && IntSize != 8 && IntSize != 16 && IntSize != 32
             && IntSize != 64 && IntSize != 128;
    }
    return false;
  });
}

bool LoopRewriteIV::run(Loop &L,
                        IVUsers *IU,
                        ScalarEvolution *SE,
                        DominatorTree *DT) {
  bool Changed = false;

  for (auto &UI : *IU) {
    // Compute the final addrec to expand into code.
    const SCEV *AR = IU->getReplacementExpr(UI);

    // Evaluate the expression out of the loop, if possible.
    if (!L.contains(UI.getUser())) {
      const SCEV *ExitVal = SE->getSCEVAtScope(AR, L.getParentLoop());
      if (SE->isLoopInvariant(ExitVal, &L))
        AR = ExitVal;
    }

    // This expression won't be acceptable for the backend.
    if (isIntWithPowerOfTwoByteSize(AR)) {
      revng_log(Log,
                "The expression has non-power-of-two integer: "
                  << dumpToString(AR));
      continue;
    }

    revng_log(Log, "Safe expression: " << dumpToString(AR));

    Value *Op = UI.getOperandValToReplace();
    Type *UseTy = Op->getType();
    Instruction *User = UI.getUser();

    revng_log(Log, "Operand: " << dumpToString(Op));
    revng_log(Log, "User Of the value to be replaced: " << dumpToString(User));

    // Determine the insertion point for this user. By default, insert
    // immediately before the user. The SCEVExpander class will automatically
    // hoist loop invariants out of the loop. For PHI nodes, there may be
    // multiple uses, so compute the nearest common dominator for the
    // incoming blocks.
    Instruction *InsertPt = getInsertPointForUses(User, Op, DT, &LI);

    // Now expand it into actual Instructions and patch it into place.
    const DataLayout &DL = L.getHeader()->getModule()->getDataLayout();
    SCEVExpander Rewriter(*SE,
                          DL,
                          "loop-rewrite-with-canonical-induction-variable");

    if (!Rewriter.isSafeToExpandAt(AR, InsertPt)) {
      revng_log(Log, "Not safe expression: " << dumpToString(AR));
      continue;
    }

    Value *NewVal = Rewriter.expandCodeFor(AR, UseTy, InsertPt);

    revng_log(Log,
              "INDVARS: Rewrote IV `" << dumpToString(AR) << "` "
                                      << dumpToString(Op) << '\n'
                                      << "   into = " << dumpToString(NewVal));

    if (NewVal != Op) {
      // Inform ScalarEvolution that this value is changing. The change doesn't
      // affect its value, but it does potentially affect which use lists the
      // value will be on after the replacement, which affects ScalarEvolution's
      // ability to walk use lists and drop dangling pointers when a value is
      // deleted.
      SE->forgetValue(User);

      // Patch the new value into place.
      if (Op->hasName())
        NewVal->takeName(Op);
      if (Instruction *NewValI = dyn_cast<Instruction>(NewVal))
        NewValI->setDebugLoc(User->getDebugLoc());
      User->replaceUsesOfWith(Op, NewVal);
      UI.setOperandValToReplace(NewVal);

      Changed = true;

      // NOTE: The old IV is a dead value, so it will be deleted with `-dce`.
    }
  }

  return Changed;
}

struct LoopRewriteWithCanonicalIVPass : public LoopPass {
public:
  static char ID;

  LoopRewriteWithCanonicalIVPass() : LoopPass(ID) {}

  bool runOnLoop(Loop *L, LPPassManager &LPM) override {
    auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    auto &IU = getAnalysis<IVUsersWrapperPass>().getIU();
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();

    auto &TheTargetTransformInfo = getAnalysis<TargetTransformInfoWrapperPass>()
                                     .getTTI(*L->getHeader()->getParent());

    LoopRewriteIV LRIV(LI);

    if (LRIV.run(*L, &IU, &SE, &DT)) {
      revng_assert(DT.verify(DominatorTree::VerificationLevel::Full));

      // Clean up the code after induction variable canonicalization.
      SmallVector<WeakTrackingVH, 16> DeadInsts;
      simplifyLoopIVs(L, &SE, &DT, &LI, &TheTargetTransformInfo, DeadInsts);
      while (!DeadInsts.empty()) {
        Value *V = DeadInsts.pop_back_val();
        if (Instruction *Inst = dyn_cast_or_null<Instruction>(V))
          RecursivelyDeleteTriviallyDeadInstructions(Inst);
      }

      for (BasicBlock *BB : L->getBlocks())
        SimplifyInstructionsInBlock(BB);

      return true;
    }

    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<IVUsersWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    getLoopAnalysisUsage(AU);
  }
};

char LoopRewriteWithCanonicalIVPass::ID = 0;
using Register = RegisterPass<LoopRewriteWithCanonicalIVPass>;
static Register X("loop-rewrite-with-canonical-induction-variable",
                  "Pass that simplifies loop induction variables by rewriting "
                  "them in a canonical form.",
                  false,
                  false);
