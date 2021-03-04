/// \file TypeShrinking.cpp
/// \brief This analysis finds which bits of each Instruction is alive

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

#include "revng/TypeShrinking/BitLiveness.h"
#include "revng/TypeShrinking/DataFlowGraph.h"
#include "revng/TypeShrinking/MFP.h"
#include "revng/TypeShrinking/TypeShrinking.h"

using namespace llvm;

using BitSet = std::set<int>;
using RegisterTypeShrinking = RegisterPass<TypeShrinking::TypeShrinking>;

static cl::opt<uint32_t> MinimumWidth("min-width",
                                      cl::init(8),
                                      cl::desc("ignore analysis results for "
                                               "width lower than"),
                                      cl::value_desc("min-width"),
                                      cl::cat(MainCategory));

char TypeShrinking::TypeShrinking::ID = 0;

static RegisterTypeShrinking
  X("type-shrinking", "Run the type shrinking analysis", true, true);

namespace TypeShrinking {

void TypeShrinking::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<BitLivenessWrapperPass>();
}

/// Returns true if each bit B of the result of Ins depends only on
/// the bits of the operands with an index lower than B
static bool isAddLike(const Instruction *Ins) {
  switch (Ins->getOpcode()) {
  case llvm::Instruction::And:
  case llvm::Instruction::Xor:
  case llvm::Instruction::Or:
  case llvm::Instruction::Add:
  case llvm::Instruction::Sub:
  case llvm::Instruction::Mul:
    return true;
  }
  return false;
}

bool TypeShrinking::runOnFunction(Function &F) {

  auto &BitLiveness = getAnalysis<BitLivenessWrapperPass>();
  auto &FixedPoints = BitLiveness.getResult();
  bool HasChanges = false;

  const std::array<uint32_t, 4> Ranks = { 8, 16, 32, 64 };
  for (auto &[Label, Result] : FixedPoints) {
    auto *Ins = Label->Instruction;
    // Find the closest rank that contains all the alive bits.
    // If there is a known rank and this is an instruction that behaves like add
    // (the least significant bits of the result depend only on the least
    // significant bits of the operands) we can down cast the operands and then
    // upcast the result
    if (Result.OutValue >= MinimumWidth.getValue() && isAddLike(Ins)) {
      auto ClosestRank = std::lower_bound(Ranks.begin(),
                                          Ranks.end(),
                                          Result.OutValue);
      if (ClosestRank != Ranks.end()
          && Ins->getType()->getScalarSizeInBits() > *ClosestRank) {
        auto Rank = *ClosestRank;
        HasChanges = true;
        llvm::Value *NewIns = nullptr;

        llvm::IRBuilder<> BuilderPre(Ins);
        llvm::IRBuilder<> BuilderPost(Ins->getNextNode());

        using CastOps = llvm::Instruction::CastOps;
        auto *Lhs = BuilderPre.CreateCast(CastOps::Trunc,
                                          Ins->getOperand(0),
                                          BuilderPre.getIntNTy(Rank));
        auto *Rhs = BuilderPre.CreateCast(CastOps::Trunc,
                                          Ins->getOperand(1),
                                          BuilderPre.getIntNTy(Rank));

        NewIns = BuilderPost.CreateBinOp((Instruction::BinaryOps)
                                           Ins->getOpcode(),
                                         Lhs,
                                         Rhs);
        auto *UpCasted = BuilderPost.CreateCast(Instruction::CastOps::ZExt,
                                                NewIns,
                                                Ins->getType());
        Ins->replaceAllUsesWith(UpCasted);
        Ins->eraseFromParent();
      }
    }
  }

  return HasChanges;
}

} // namespace TypeShrinking
