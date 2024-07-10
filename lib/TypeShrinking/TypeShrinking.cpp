/// \file TypeShrinking.cpp
/// This analysis finds which bits of each Instruction is alive.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"

#include "revng/Support/CommandLine.h"
#include "revng/Support/IRHelpers.h"
#include "revng/TypeShrinking/BitLiveness.h"
#include "revng/TypeShrinking/DataFlowGraph.h"
#include "revng/TypeShrinking/TypeShrinking.h"

using namespace llvm;

using BitSet = std::set<int>;

static cl::opt<uint32_t> MinimumWidth("min-width",
                                      cl::init(8),
                                      cl::desc("ignore analysis results for "
                                               "width lower than"),
                                      cl::value_desc("min-width"),
                                      cl::cat(MainCategory));

char TypeShrinking::TypeShrinkingWrapperPass::ID = 0;

using Register = RegisterPass<TypeShrinking::TypeShrinkingWrapperPass>;
static Register
  X("type-shrinking", "Run the type shrinking analysis", true, true);

namespace TypeShrinking {

void TypeShrinkingWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<BitLivenessWrapperPass>();
}

/// Returns true if each bit B of the result of Ins depends only on the bits of
/// the operands with an index equal to B
static bool isBitwise(const Instruction *Ins) {
  switch (Ins->getOpcode()) {
  case llvm::Instruction::And:
  case llvm::Instruction::Xor:
  case llvm::Instruction::Or:
    return true;
  }
  return false;
}

/// Returns true if each bit B of the result of Ins depends only on the bits of
/// the operands with an index lower than B
static bool isAddLike(const Instruction *Ins) {
  switch (Ins->getOpcode()) {
  case llvm::Instruction::Add:
  case llvm::Instruction::Sub:
  case llvm::Instruction::Mul:
  case llvm::Instruction::Shl:
  case llvm::Instruction::Select:
  case llvm::Instruction::PHI:
    return true;
  }
  return false;
}

static bool runTypeShrinking(Function &F,
                             const BitLivenessAnalysisResults &FixedPoints) {
  bool HasChanges = false;

  IRBuilder<> B(F.getParent()->getContext());
  const std::array<uint32_t, 4> Ranks = { 8, 16, 32, 64 };
  for (auto &[I, Result] : FixedPoints) {
    // Find the closest rank that contains all the alive bits.
    // If there is a known rank and this is an instruction that behaves like add
    // (the least significant bits of the result depend only on the least
    // significant bits of the operands) we can down cast the operands and then
    // upcast the result
    if (I->getType()->isIntegerTy()
        and (isBitwise(I) or isAddLike(I) or isa<ICmpInst>(I))) {
      // Bound analysis results to MinimumWidth
      unsigned NewResultSize = std::max(MinimumWidth.getValue(), Result.Result);
      unsigned NewOperandsSize = std::max(MinimumWidth.getValue(),
                                          Result.Operands);

      // Get old size
      Type *OldType = I->getType();
      unsigned OldSize = OldType->getIntegerBitWidth();

      // Find closest rank
      auto It = llvm::lower_bound(Ranks, NewOperandsSize);
      if (It == Ranks.end())
        NewOperandsSize = OldSize;
      else
        NewOperandsSize = *It;
      Type *NewOperandsType = B.getIntNTy(NewOperandsSize);

      if (isBitwise(I)) {
        NewResultSize = NewOperandsSize;
      } else {
        It = llvm::lower_bound(Ranks, NewResultSize);
        if (It == Ranks.end())
          NewResultSize = OldSize;
        else
          NewResultSize = *It;
      }
      Type *NewResultType = B.getIntNTy(NewResultSize);

      if (NewOperandsSize > NewResultSize)
        NewResultSize = NewOperandsSize;

      if (NewOperandsSize < OldSize) {
        B.SetInsertPoint(I);

        // Shrink operands
        Value *LHS = B.CreateTrunc(I->getOperand(0), NewOperandsType);
        Value *RHS = B.CreateTrunc(I->getOperand(1), NewOperandsType);

        // Recreate instruction
        LHS = B.CreateZExt(LHS, NewResultType);
        RHS = B.CreateZExt(RHS, NewResultType);
        auto Opcode = static_cast<Instruction::BinaryOps>(I->getOpcode());
        Value *Result = B.CreateBinOp(Opcode, LHS, RHS);

        // Emit ZExts, as late as possible
        SmallVector<std::pair<Use *, Value *>, 6> Replacements;
        for (Use &TheUse : I->uses()) {
          if (auto *I = cast<Instruction>(TheUse.getUser())) {
            B.SetInsertPoint(I);

            // Fix insert point for PHIs
            if (auto *Phi = dyn_cast<PHINode>(I)) {
              auto *BB = Phi->getIncomingBlock(TheUse);
              auto It = BB->getTerminator()->getIterator();
              B.SetInsertPoint(BB, It);
            }

            auto *LateUpcast = B.CreateZExt(Result, OldType);
            Replacements.emplace_back(&TheUse, LateUpcast);
          }
        }

        // Apply replacements
        for (auto &[Use, I] : Replacements)
          Use->set(I);

        // Drop the original instruction
        eraseFromParent(I);
      }
    }
  }

  //
  // Drop zext from zext(value) == constant
  //
  SmallVector<ICmpInst *, 8> Compares;
  for (Instruction &I : instructions(&F))
    if (auto *ICmp = dyn_cast<ICmpInst>(&I))
      if (ICmp->getPredicate() == llvm::CmpInst::Predicate::ICMP_EQ)
        Compares.push_back(ICmp);

  for (ICmpInst *ICmp : Compares) {
    // Pattern match zext(value) == constant
    auto *ZEXt = dyn_cast<ZExtInst>(ICmp->getOperand(0));
    auto *RHS = dyn_cast<ConstantInt>(ICmp->getOperand(1));
    if (ZEXt == nullptr or RHS == nullptr)
      continue;

    Value *PreExtension = ZEXt->getOperand(0);
    auto *PreExtensionType = dyn_cast<IntegerType>(PreExtension->getType());
    if (PreExtensionType == nullptr)
      continue;

    unsigned PreExtensionSize = PreExtensionType->getIntegerBitWidth();
    unsigned PostExtensionSize = ZEXt->getType()->getIntegerBitWidth();
    APInt TruncatedRHS = RHS->getValue().trunc(PreExtensionSize);

    // Ensure the upper bits of the constant are zero
    unsigned ExpectedLeadingZeros = PostExtensionSize - PreExtensionSize;
    if (RHS->getValue().countLeadingZeros() < ExpectedLeadingZeros)
      continue;

    // Replace the compare trunc'ing the constant and skipping over the zext
    HasChanges = true;
    IRBuilder<> B(ICmp);
    auto *NewICmp = B.CreateICmp(ICmp->getPredicate(),
                                 PreExtension,
                                 ConstantInt::get(PreExtensionType,
                                                  TruncatedRHS));
    ICmp->replaceAllUsesWith(NewICmp);
    ICmp->eraseFromParent();
  }

  return HasChanges;
}

bool TypeShrinkingWrapperPass::runOnFunction(Function &F) {
  auto &BitLiveness = getAnalysis<BitLivenessWrapperPass>();
  auto &FixedPoints = BitLiveness.getResult();
  return runTypeShrinking(F, FixedPoints);
}

PreservedAnalyses TypeShrinkingPass::run(Function &F,
                                         FunctionAnalysisManager &FAM) {
  const auto &FixedPoints = FAM.getResult<BitLivenessPass>(F);
  bool HasChanges = runTypeShrinking(F, FixedPoints);
  return HasChanges ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

} // namespace TypeShrinking
