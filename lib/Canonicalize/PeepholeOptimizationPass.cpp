//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"

#include "revng/Support/Debug.h"
#include "revng/Support/IRBuilder.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

static Logger Log("peephole-opt-for-decompilation");

struct PeepholeOptimizationPass : public FunctionPass {
public:
  static char ID;

  PeepholeOptimizationPass() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

  bool runOnFunction(Function &F) override;
};

// This function looks at the incoming values for PHI, trying to rewrite them
// in a way that they have more uses.
//  * in general I don't know
//  * in practice, the current implementation rewrites incomings that are addsub
//    with constants and that are used in icmp with constants, so that the icmp
//    itself is changed to use the AddSub non-const operand, and the constants
//    are folded. So you go from e.g. x + C1 == c2 to x == C1 - c2 (and then
//    C1-c2) is folded I guess. So the PHINode still uses x + C1, but there are
//    more instructions using x directly.
//    This is beneficial in particular for comparisons used to break out of
//    loops, which in decompilation would look awkward having comparison like
//    x + C1 == c2.
//  * actually, if the PHINode is the "induction variable", what we really want
//    is that we increase the number of uses of the PHINode, in practice
//    changing things in a way that more instructions use the PHINode directly
//    instead of one of its incomings, when inside a loop.
//    in order for this to be possible we need:
//    * a) that the PHINode dominates the incoming I
//    * b) that the incoming I uses the PHINode as one of its operands
//    * c) that some of the users of I can be rewritten to use PHINode directly,
//         instead of the intermediate value I
//    There are mainly 2 scenarios when this can happen:
//    1) I is an AddSub with a constant PHI + C1.
//      1.a) One of I's users is also an AddSub with another constants I + c2.
//           They can be folded into PHI + (C1 + c2), as long as PHI dominates I
//           + c2.
//      1.b) One of I's users is a Eq/Neq wht constant I != c2.
//           They can be folded into PHI != (C1 + c2), as long as PHI dominates
//           I != c2.
static bool reusePHIIncomings(PHINode &PHI, const DominatorTree &DT) {
  // Ignore non-integer PHINodes, since all the patterns we're capable of
  // handling are on integers.
  if (not PHI.getType()->isIntegerTy())
    return false;

  // Here we should definitely use the builder that checks the debug info,
  // but since this going to go away soon, let it stay as is.
  revng::IRBuilder Builder(PHI.getContext());

  revng_log(Log, "PHI: " << dumpToString(PHI));
  revng_log(Log, "incomings: ");
  LoggerIndent Indent{ Log };

  bool Changed = false;
  for (Use &IncomingUse : PHI.incoming_values()) {
    // TODO: we only look at Add and Sub for now, since they are the only ones
    // whose rewrite seemed beneficial in the real world examples we looked at.
    // There might be other opportunities that we're missing, possibly all
    // invertible binary operators whose RHS is a constant.
    auto *AddSub = dyn_cast<BinaryOperator>(IncomingUse.get());
    if (not AddSub)
      continue;

    if (unsigned OpCode = AddSub->getOpcode();
        OpCode != Instruction::Add and OpCode != Instruction::Sub)
      continue;

    // Only consider AddSub if the LHS not constant and RHS is constant.
    if (not isa<Constant>(AddSub->getOperand(1))
        or isa<Constant>(AddSub->getOperand(0)))
      continue;

    revng_log(Log,
              "Found AddSub with non-constant LHS and constant RHS: "
                << dumpToString(AddSub));

    // At this point we've found AddSub to be an incoming value for PHI that is
    // an Add or Sub whose LHS is non constant and whose RHS is a constant. We
    // want to try and rewrite as many instructions as possible to reuse AddSub
    // instead of the incoming of the PHI, when possible.

    // AddSub is in the form X +- C1 (where X is not a constant, and C1 is)
    // Let's look at other ICmp instructions in the form `X == C2` or `X != C2`,
    // so we can rewrite them as `AddSub == C2 +- C1` or `AddSub != C2 +- C1`
    // The rewrite is valid if either of the following conditions applies:
    // * AddSub dominates ICmp
    // * ICmp dominates AddSub, in which case we also have to anticipate AddSub
    //   right before ICmp, so ICmp can be rewritten to use AddSub.
    // We can do this in steps.
    // 1. we look at all uses of X that are ICmp with constants that respect the
    //   dominance properties above.
    // 2. we pick the best place where AddSub can be anticipated.
    // 3. we anticipate AddSub. this is always valid because the RHS of both
    //    AddSub and ICmp is a constant, and the LHS is the same
    // 4. now AddSub dominates all candidate ICmp that match.
    // 5. we rewrite all of them.
    // NOTE: we can also rewrite ICmp that don't dominate or are dominated by
    // AddSub, by moving AddSub to their common dominator, but that's not
    // beneficial for recompilation. If anything it's harmful.

    Value *X = AddSub->getOperand(0);
    auto *C1 = cast<Constant>(AddSub->getOperand(1));

    const auto IsRewritableICmp = [X](User *I) -> ICmpInst * {
      if (auto *ICMP = dyn_cast<ICmpInst>(I);
          ICMP and ICMP->isEquality() and isa<Constant>(I->getOperand(1))
          and I->getOperand(0) == X)
        return ICMP;
      return nullptr;
    };

    // 1. + 2.
    SmallVector<ICmpInst *> RewritableICmp;
    Instruction *CommonDominator = AddSub;
    for (User *TheUser : X->users()) {
      if (ICmpInst *ICmp = IsRewritableICmp(TheUser)) {
        revng_log(Log, "Found rewritable ICmp: " << dumpToString(ICmp));
        if (DT.dominates(AddSub, ICmp)) {
          revng_log(Log, "AddSub dominates ICmp");
          RewritableICmp.push_back(ICmp);
        } else if (DT.dominates(ICmp, AddSub)) {
          revng_log(Log, "AddSub dominates ICmp");
          RewritableICmp.push_back(ICmp);
          CommonDominator = ICmp;
        }
      }
    }

    // 3.
    if (CommonDominator != AddSub) {
      AddSub->removeFromParent();
      AddSub->insertBefore(CommonDominator);
      revng_log(Log, "Move AddSub before CommonDominator");
    }
    // 4.
    // 5.
    for (ICmpInst *ToRewrite : RewritableICmp) {

      // Now we have to figure out the opcodes of the rewritten expression for
      // ICmp, depending on the opcodes of AddSub and ICmp.
      //   AddSub -> ||       X + C1        |       X - C1        |
      //             ||                     |                     |
      // | ICmp      ||                     |                     |
      // v           ||                     |                     |
      // ======== ===++=====================+=====================+
      //  X == C2    || AddSub == (C2 + C1) | AddSub == (C2 - C1) |
      //  X != C2    || AddSub != (C2 + C1) | AddSub != (C2 - C1) |
      //
      // So the opcode of the addsub to the RHS of the comparison is always the
      // same as the AddSub.

      // Build the new operation among constants
      Builder.SetInsertPoint(ToRewrite);
      auto *C2 = cast<Constant>(ToRewrite->getOperand(1));
      auto *NewRHS = Builder.CreateBinOp(AddSub->getOpcode(), C2, C1);

      // If it's an ICmp we can replace the operands in place.
      ToRewrite->setOperand(0, AddSub);
      ToRewrite->setOperand(1, NewRHS);
      revng_log(Log, "Rewritten ICmp: " << dumpToString(ToRewrite));
      Changed = true;
    }
  }

  return Changed;
}

bool PeepholeOptimizationPass::runOnFunction(Function &F) {
  revng_log(Log, "Peephole For Decompilation: " << F.getName());
  LoggerIndent Indent{ Log };
  bool Changed = false;
  DominatorTree DT;
  DT.recalculate(F);
  for (BasicBlock &B : F) {
    for (PHINode &PHI : B.phis()) {
      Changed |= reusePHIIncomings(PHI, DT);
    }
  }
  return Changed;
}

char PeepholeOptimizationPass::ID = 0;

using Register = RegisterPass<PeepholeOptimizationPass>;
static Register
  X("peephole-opt-for-decompilation", "PeepholeOptimizationPass", false, false);
