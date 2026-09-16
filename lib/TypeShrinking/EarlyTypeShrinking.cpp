/// A set of simple transformation that shrink the type of certain instructions.
/// This should be run before TypeShrinking.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"

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

/// The width the pass is willing to truncate to
static bool isEmittableWidth(uint64_t Size) {
  return Size == 8 or Size == 16 or Size == 32 or Size == 64;
}

/// Rewrite a left shift followed by a right shift as the cast it stands for.
///
/// The pattern is `(x << A) >> B`, where `A` and `B` are the two constant
/// amounts (`LeftShiftAmount` and `RightShiftAmount` below), and `x`, both
/// shifts and the result are all of one integer type, `OuterSize` bits wide.
///
/// `x << A` keeps only the low `OuterSize - A` bits of `x`, so those are the
/// only ones the pair as a whole looks at: call that width `InnerSize` and
/// write `T` for `x` truncated to it. What the pair computes is `T`, extended
/// back to `OuterSize` and shifted by whatever the two amounts leave over.
///
/// The extension is the one the right shift performs. `ashr` copies the top
/// bit of the value it shifts, which is the top bit of `T`, so the pair sign
/// extends `T`; `lshr` shifts zeroes in, so it zero extends it.
///
///     ; B > A: `T`, shifted down by the difference, then extended
///     %0 = shl  i64 %x, 32
///     %1 = lshr i64 %0, 35     ; zext(lshr(trunc %x to i32, 3)) to i64
///
///     ; B == A: nothing left over, a plain cast
///     %0 = shl  i64 %x, 32
///     %1 = ashr i64 %0, 32     ; sext(trunc %x to i32) to i64
///
///     ; B < A: that cast, shifted back up by the difference
///     %0 = shl  i64 %x, 32
///     %1 = ashr i64 %0, 29     ; shl(sext(trunc %x to i32) to i64, 3)
///
/// The last one is the only one that needs an argument, since it shifts left
/// after extending rather than before. The extension fills the `A` bits above
/// `T` with copies of one bit (the top bit of `T`, or zero), and shifting
/// left by `A - B` discards `A - B` of them, which is at most `A`, so nothing
/// but copies is lost.
static bool shrinkShiftPair(revng::IRBuilder &B, Instruction &I) {
  using namespace PatternMatch;

  uint64_t RightShiftAmount = 0;
  uint64_t LeftShiftAmount = 0;
  llvm::Value *Value = nullptr;

  // Match (x << A) >> B
  if (not match(&I,
                m_Shr(m_Shl(m_Value(Value), m_ConstantInt(LeftShiftAmount)),
                      m_ConstantInt(RightShiftAmount)))) {
    return false;
  }

  Type *OuterType = I.getType();
  uint64_t OuterSize = OuterType->getIntegerBitWidth();

  // A shift by at least the width is poison; leave it alone rather than
  // underflow `InnerSize`.
  if (LeftShiftAmount >= OuterSize or RightShiftAmount >= OuterSize)
    return false;

  uint64_t InnerSize = OuterSize - LeftShiftAmount;
  if (not isEmittableWidth(InnerSize))
    return false;

  auto *InnerType = IntegerType::get(I.getContext(), InnerSize);

  B.SetInsertPoint(&I);
  llvm::Value *Truncated = B.CreateTrunc(Value, InnerType);

  const bool IsArithmetic = I.isArithmeticShift();

  if (RightShiftAmount > LeftShiftAmount) {
    uint64_t InnerRightShift = RightShiftAmount - LeftShiftAmount;
    Truncated = IsArithmetic ? B.CreateAShr(Truncated, InnerRightShift) :
                               B.CreateLShr(Truncated, InnerRightShift);
  }

  llvm::Value *Replacement = IsArithmetic ? B.CreateSExt(Truncated, OuterType) :
                                            B.CreateZExt(Truncated, OuterType);

  if (RightShiftAmount < LeftShiftAmount)
    Replacement = B.CreateShl(Replacement, LeftShiftAmount - RightShiftAmount);

  I.replaceAllUsesWith(Replacement);
  return true;
}

/// Rewrite a comparison between two values shifted left by the same amount as
/// a comparison between the bits that shift keeps.
///
/// The pattern is `icmp <Pred> (x << K), (y << K)`, where `K` is the constant
/// amount both sides are shifted by and `x`, `y` and the two shifts are all of
/// one integer type, `OuterSize` bits wide. As in `shrinkShiftPair`, `x << K`
/// carries only the low `OuterSize - K` bits of `x`, so the comparison is one
/// between the two operands truncated to that width, `InnerSize`:
///
///     ; a 32-bit signed comparison between two 64-bit registers
///     %0 = shl i64 %x, 32
///     %1 = shl i64 %y, 32
///     %2 = icmp slt i64 %0, %1     ; icmp slt i32 trunc %x, trunc %y
///
/// Every predicate survives the rewrite. Read as an unsigned number, `x << K`
/// is `zext(trunc x)` times `2^K`; read as a signed one, it is `sext(trunc x)`
/// times `2^K`, and neither product overflows `OuterSize` bits. Multiplying
/// both sides by the same positive constant leaves the signed order, the
/// unsigned order and equality alone.
///
/// Unlike `shrinkShiftPair` there is no right shift here to turn into a cast,
/// so without this the comparison reaches the backend as a pair of shifts with
/// an `icmp` between them.
static bool shrinkShiftedCompare(revng::IRBuilder &B, Instruction &I) {
  using namespace PatternMatch;

  auto *Compare = dyn_cast<ICmpInst>(&I);
  if (Compare == nullptr)
    return false;

  llvm::Value *LHS = nullptr;
  llvm::Value *RHS = nullptr;
  uint64_t LHSAmount = 0;
  uint64_t RHSAmount = 0;

  if (not match(Compare->getOperand(0),
                m_Shl(m_Value(LHS), m_ConstantInt(LHSAmount))))
    return false;
  if (not match(Compare->getOperand(1),
                m_Shl(m_Value(RHS), m_ConstantInt(RHSAmount))))
    return false;

  if (LHSAmount != RHSAmount or LHSAmount == 0)
    return false;

  Type *OperandType = LHS->getType();
  if (not OperandType->isIntegerTy())
    return false;

  uint64_t OuterSize = OperandType->getIntegerBitWidth();
  if (LHSAmount >= OuterSize)
    return false;

  uint64_t InnerSize = OuterSize - LHSAmount;
  if (not isEmittableWidth(InnerSize))
    return false;

  auto *InnerType = IntegerType::get(I.getContext(), InnerSize);

  B.SetInsertPoint(&I);
  llvm::Value *Replacement = B.CreateICmp(Compare->getPredicate(),
                                          B.CreateTrunc(LHS, InnerType),
                                          B.CreateTrunc(RHS, InnerType));

  I.replaceAllUsesWith(Replacement);
  return true;
}

bool EarlyTypeShrinking::runOnFunction(Function &F) {
  bool Changed = false;

  // TODO: checks are only omitted here because of unit tests.
  revng::IRBuilder B(F.getContext());

  for (Instruction &I : llvm::make_early_inc_range(llvm::instructions(F))) {
    if (shrinkShiftPair(B, I) or shrinkShiftedCompare(B, I))
      Changed = true;
  }

  return Changed;
}

RegisterPass<EarlyTypeShrinking> Y("early-type-shrinking",
                                   "Preliminary instruction type shrinking",
                                   true,
                                   true);
