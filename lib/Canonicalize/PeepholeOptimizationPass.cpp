//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"

#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

static Logger<> Log("peephole-opt-for-decompilation");

struct PeepholeOptimizationPass : public FunctionPass {
public:
  static char ID;

  PeepholeOptimizationPass() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

  bool runOnFunction(Function &F) override;
};

static bool isICMPWithConstRHS(const Instruction &I) {
  if (auto *ICMP = dyn_cast<ICmpInst>(&I))
    return ICMP->isEquality() and isa<ConstantInt>(I.getOperand(1));
  return false;
}

static bool replaceNonConstantOperandWithAddSub(Instruction &I,
                                                BinaryOperator *AddSub,
                                                IRBuilder<> &Builder) {
  if (not isICMPWithConstRHS(I))
    return false;

  auto *ICmp = cast<ICmpInst>(&I);

  auto *ICmpNonConstOp = ICmp->getOperand(0);
  auto *ICmpConstOp = cast<ConstantInt>(ICmp->getOperand(1));

  auto *AddSubNonConstantOp = AddSub->getOperand(0);
  auto *AddSubConstantOp = cast<ConstantInt>(AddSub->getOperand(1));

  if (ICmpNonConstOp != AddSubNonConstantOp)
    return false;

  revng_log(Log, "Found I: " << dumpToString(I));

  // Now we have to figure out the opcodes of the rewritten expression for I,
  // depending on the opcodes of AddSub and I.
  // In the table below, x is the Incominv value of the PHI, and we want to
  // rewrite I as explained in the table, so that it doesn't use x anymore, and
  // it uses AddSub instead.
  //   AddSub-> ||       x + c1        |       x - c1        |
  // I          ||                     |                     |
  // |          ||                     |                     |
  // v          ||                     |                     |
  // ===========++=====================+=====================+
  //  x == c2   || AddSub == (c2 + c1) | AddSub == (c2 - c1) |
  //  x != c2   || AddSub != (c2 + c1) | AddSub != (c2 - c1) |

  Instruction::BinaryOps AddSubOpCode = AddSub->getOpcode();
  revng_assert(AddSubOpCode == Instruction::Add
               or AddSubOpCode == Instruction::Sub);

  // Build the new arithmetic among constants
  Builder.SetInsertPoint(&I);

  // Build the new operation among constants
  auto *ConstRHS = AddSubConstantOp;
  auto *ConstLHS = ICmpConstOp;
  auto *NewConstOp = Builder.CreateBinOp(AddSubOpCode, ConstLHS, ConstRHS);

  // If it's an ICmp we can replace the operands in place.
  ICmp->setOperand(0, AddSub);
  ICmp->setOperand(1, NewConstOp);

  return true;
}

static bool reusePHIIncomings(PHINode &PHI, const DominatorTree &DT) {
  // Ignore non-integers
  if (not PHI.getType()->isIntegerTy())
    return false;

  IRBuilder<> Builder(PHI.getContext());

  revng_log(Log, "Decompilation: " << dumpToString(PHI));
  revng_log(Log, "uses: ");
  LoggerIndent Indent{ Log };

  bool Changed = false;
  for (Use &IncomingUse : PHI.incoming_values()) {
    Value *Incoming = IncomingUse.get();
    BasicBlock *IncomingBlock = PHI.getIncomingBlock(IncomingUse);
    // Only look at binary operators
    auto *AddSub = dyn_cast<BinaryOperator>(Incoming);
    if (not AddSub)
      continue;

    // TODO: we only look at Add and Sub, since they are the only obviously
    // invertible ones.
    // There might be other opportunities that we're missing.
    unsigned OpCode = AddSub->getOpcode();
    if (OpCode != Instruction::Add and OpCode != Instruction::Sub)
      continue;

    // Assume we're in instcombine form. Check if the RHS is constant.
    auto *AddSubConstantOp = dyn_cast<ConstantInt>(AddSub->getOperand(1));
    if (not AddSubConstantOp or isa<Constant>(AddSub->getOperand(0)))
      continue;

    revng_log(Log, "Found AddSub with const operand: " << dumpToString(AddSub));

    // At this point we've found an incoming value for PHI that is an Add or Sub
    // and whose RHS is a constant.
    // We want to try and reduce the uses of Incoming.
    // To do that, we want to see if there is any other instruction dominated by
    // Incoming that can be rewritten to reuse Incoming instead of some other
    // operands.

    BasicBlock *AddSubBlock = AddSub->getParent();

    revng_log(Log, "Look for instruction that can be rewritten using AddSub");
    LoggerIndent MoreIndent{ Log };
    // Start looking in remainder of the same block of Incoming.
    {
      revng_log(Log, "In Block: " << AddSubBlock->getName());
      LoggerIndent MoreIndent{ Log };
      auto AfterIncoming = std::next(AddSub->getIterator());
      auto BinaryOpBlockEnd = AddSubBlock->getTerminator()->getIterator();
      for (Instruction &I : llvm::make_range(AfterIncoming, BinaryOpBlockEnd))
        Changed |= replaceNonConstantOperandWithAddSub(I, AddSub, Builder);
    }

    // Then look at all the other blocks dominated by Incoming.
    for (auto *DominatorNode : llvm::depth_first(DT.getNode(IncomingBlock))) {
      BasicBlock *BB = DominatorNode->getBlock();
      if (BB == AddSubBlock)
        continue;

      revng_log(Log, "In Block: " << BB->getName());
      LoggerIndent MoreIndent{ Log };

      for (Instruction &I : *BB)
        Changed |= replaceNonConstantOperandWithAddSub(I, AddSub, Builder);
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
