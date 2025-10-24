/// \file EarlyTypeShrinking.cpp
/// A set of simple transformation that shrink the type of certain instructions.
/// This should be run before TypeShrinking.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/InstIterator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"

#include "revng/Support/Assert.h"
#include "revng/Support/IRBuilder.h"

using namespace llvm;

class EarlyTypeShrinking : public llvm::FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  EarlyTypeShrinking() : FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {}
};

char EarlyTypeShrinking::ID = 0;

bool EarlyTypeShrinking::runOnFunction(Function &F) {
  bool Changed = false;

  // TODO: checks are only omitted here because of unit tests.
  revng::NonDebugInfoCheckingIRBuilder B(F.getContext());

  for (Instruction &I : llvm::make_early_inc_range(llvm::instructions(F))) {
    using namespace PatternMatch;

    uint64_t RightShiftAmount = 0;
    uint64_t LeftShiftAmount = 0;
    llvm::Value *Value = nullptr;

    // Match (x << a) >> b
    if (not match(&I,
                  m_Shr(m_Shl(m_Value(Value), m_ConstantInt(LeftShiftAmount)),
                        m_ConstantInt(RightShiftAmount)))) {
      continue;
    }

    if (RightShiftAmount <= LeftShiftAmount)
      continue;

    Type *OuterType = I.getType();
    uint64_t OuterSize = OuterType->getIntegerBitWidth();
    uint64_t InnerSize = OuterSize - LeftShiftAmount;
    auto *InnerType = IntegerType::get(F.getContext(), InnerSize);

    if (not isPowerOf2_64(InnerSize))
      continue;

    uint64_t InnerRightShift = RightShiftAmount - LeftShiftAmount;
    revng_assert(static_cast<int64_t>(InnerRightShift) > 0);

    B.SetInsertPoint(&I);
    llvm::Value *Replacement = nullptr;

    auto *Truncated = B.CreateTrunc(Value, InnerType);

    if (I.isArithmeticShift()) {
      Replacement = B.CreateSExt(B.CreateAShr(Truncated, InnerRightShift),
                                 OuterType);
    } else {
      Replacement = B.CreateZExt(B.CreateLShr(Truncated, InnerRightShift),
                                 OuterType);
    }

    I.replaceAllUsesWith(cast<Instruction>(Replacement));
    Changed = true;
  }

  return Changed;
}

RegisterPass<EarlyTypeShrinking> Y("early-type-shrinking",
                                   "Preliminary instruction type shrinking",
                                   true,
                                   true);
