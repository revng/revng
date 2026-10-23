//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"

#include "revng/Canonicalize/SimplifyComparisons.h"
#include "revng/Support/IRBuilder.h"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {

/// Expose boolean predicates in packed flags and recover narrow signed tests.
/// Distribute zero comparisons over ORs, move shifted masks onto their sources,
/// and turn sign-bit tests into signed comparisons at the corresponding width.
class SimplifyComparisonsWrapperPass : public FunctionPass {
public:
  static char ID;

public:
  SimplifyComparisonsWrapperPass() : FunctionPass(ID) {}

public:
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

public:
  bool runOnFunction(Function &F) override;
};

class ComparisonSimplifier {
private:
  SmallVector<ICmpInst *, 16> WorkList;

public:
  bool run(Function &F);

private:
  bool distributeOrComparison(ICmpInst &Comparison);
  static bool canonicalizeShiftedMask(ICmpInst &Comparison);
  static bool simplifySignBitComparison(ICmpInst &Comparison);
};

/// Expose the individual predicates hidden by a zero comparison on an `or`:
///
///     (A | B) == 0  ->  (A == 0) && (B == 0)
///     (A | B) != 0  ->  (A != 0) || (B != 0)
///
/// This lets subsequent iterations simplify comparisons on A and B
/// independently. InstCombine generally prefers the reverse transformation.
/// Only expand single-use ORs so shared expressions cannot cause exponential
/// growth. Each expansion consumes its matched OR instead of duplicating it.
bool ComparisonSimplifier::distributeOrComparison(ICmpInst &Comparison) {
  Value *V = Comparison.getOperand(0);
  Value *LHSOperand = nullptr;
  Value *RHSOperand = nullptr;
  if (not match(V, m_OneUse(m_Or(m_Value(LHSOperand), m_Value(RHSOperand)))))
    return false;

  revng::IRBuilder Builder(Comparison.getContext());
  Builder.SetInsertPoint(&Comparison, Comparison.getDebugLoc());
  auto Predicate = Comparison.getPredicate();
  Value *Zero = ConstantInt::get(V->getType(), 0);
  Value *LHS = Builder.CreateICmp(Predicate, LHSOperand, Zero);
  Value *RHS = Builder.CreateICmp(Predicate, RHSOperand, Zero);
  Value *Replacement = Predicate == ICmpInst::ICMP_EQ ?
                         Builder.CreateAnd(LHS, RHS) :
                         Builder.CreateOr(LHS, RHS);

  if (auto *NewComparison = dyn_cast<ICmpInst>(LHS))
    WorkList.push_back(NewComparison);

  if (auto *NewComparison = dyn_cast<ICmpInst>(RHS))
    WorkList.push_back(NewComparison);

  if (auto *I = dyn_cast<Instruction>(Replacement))
    I->takeName(&Comparison);

  Comparison.replaceAllUsesWith(Replacement);
  Comparison.eraseFromParent();

  // Release the operands' uses so unshared OR trees can expand further.
  if (auto *Or = dyn_cast<Instruction>(V))
    Or->eraseFromParent();

  return true;
}

/// Move constant right shifts out of a masked zero comparison:
///
///     ((X >> Shift) & Mask) == 0
///         -> (X & (Mask << Shift)) == 0
///
/// The masked values are not equivalent, but their comparisons with zero are.
/// For example, ((X >> 24) & 128) != 0 becomes (X & 0x80000000) != 0.
/// The sign-bit matcher can then recognize bit 31 directly, without walking
/// the shift chain. LLVM's foldICmpAndShift could supply this fold, but it is
/// internal to InstCombinerImpl and uses its builder and worklist. Reusing it
/// would require extracting a shared helper. Running the full InstCombine pass
/// here could recombine the newly exposed predicates.
bool ComparisonSimplifier::canonicalizeShiftedMask(ICmpInst &Comparison) {
  Value *V = Comparison.getOperand(0);
  Value *Source = nullptr;
  const APInt *MatchedMask = nullptr;
  if (not match(V, m_And(m_Value(Source), m_APInt(MatchedMask))))
    return false;

  APInt Mask = *MatchedMask;
  bool Changed = false;

  while (true) {
    Value *Unshifted = nullptr;
    const APInt *MatchedAmount = nullptr;
    if (not match(Source, m_Shr(m_Value(Unshifted), m_APInt(MatchedAmount))))
      break;

    uint64_t ShiftAmount = MatchedAmount->getLimitedValue(Mask.getBitWidth());
    // The identity only holds if every observed result bit originated in the
    // value being shifted.
    if (ShiftAmount >= Mask.getBitWidth()
        or Mask.getActiveBits() > Mask.getBitWidth() - ShiftAmount) {
      return false;
    }

    Source = Unshifted;
    Mask <<= ShiftAmount;
    Changed = true;
  }

  if (not Changed)
    return false;

  revng::IRBuilder Builder(Comparison.getContext());
  Builder.SetInsertPoint(&Comparison, Comparison.getDebugLoc());
  Value *MaskedValue = Builder.CreateAnd(Source,
                                         ConstantInt::get(Source->getType(),
                                                          Mask),
                                         "compared.mask");
  Comparison.setOperand(0, MaskedValue);
  return true;
}

/// Turn a test of the sign bit of a supported integer width into a signed
/// comparison. Shifted masks have already been canonicalized, so for example:
///
///     %shifted = lshr i64 %x, 24
///     %sign = and i64 %shifted, 128
///     %negative = icmp ne i64 %sign, 0
///
/// first becomes:
///
///     %sign = and i64 %x, 0x80000000
///     %negative = icmp ne i64 %sign, 0
///
/// and is then simplified to:
///
///     %value = trunc i64 %x to i32
///     %negative = icmp slt i32 %value, 0
///
/// InstCombine recognizes the existing type's sign bit. It previously had
/// this narrowing fold, but now canonicalizes in the reverse direction.
/// Adding this fold alongside that rule would create a rewrite cycle.
bool ComparisonSimplifier::simplifySignBitComparison(ICmpInst &Comparison) {
  Value *V = Comparison.getOperand(0);
  Value *Source = nullptr;
  const APInt *Mask = nullptr;
  if (not match(V, m_And(m_Value(Source), m_APInt(Mask))))
    return false;

  auto *SourceType = dyn_cast<IntegerType>(Source->getType());
  if (not SourceType or SourceType->getBitWidth() != Mask->getBitWidth())
    return false;

  unsigned Width = Mask->getActiveBits();

  // Comparing the narrowed value against zero asks whether one single bit is
  // set, so the mask has to select exactly that bit. A mask with more bits set
  // asks a different question, which this rewrite would not answer.
  if (not Mask->isPowerOf2())
    return false;

  // The mask selects bit Width - 1, which makes Width the narrowest type whose
  // sign bit that bit is. Only a power of two is the width of a type the rest
  // of the pipeline deals in, so this leaves the bits 0, 1, 3, 7, 15, 31 and
  // 63, and turns down a mask such as 4, whose Width of 3 would ask for an i3.
  if (not isPowerOf2_64(Width))
    return false;

  // Of the widths that leaves, 2 and 4 are not whole bytes, so the backend
  // cannot emit them and the clifter aborts on one. A single bit is fine: the
  // clifter widens that one to a byte and has a path of its own for it.
  if (Width == 2 or Width == 4)
    return false;

  revng::IRBuilder Builder(Comparison.getContext());
  Builder.SetInsertPoint(&Comparison, Comparison.getDebugLoc());
  Value *Truncated = Source;
  if (Width != SourceType->getBitWidth()) {
    Truncated = Builder.CreateTrunc(Source,
                                    Builder.getIntNTy(Width),
                                    "compared.value");
  }

  ICmpInst::Predicate Predicate = Comparison.getPredicate();
  Predicate = Predicate == ICmpInst::ICMP_EQ ? ICmpInst::ICMP_SGE :
                                               ICmpInst::ICMP_SLT;

  Value *Replacement = Builder.CreateICmp(Predicate,
                                          Truncated,
                                          ConstantInt::get(Truncated->getType(),
                                                           0));
  if (auto *I = dyn_cast<Instruction>(Replacement))
    I->takeName(&Comparison);

  Comparison.replaceAllUsesWith(Replacement);
  Comparison.eraseFromParent();
  return true;
}

bool ComparisonSimplifier::run(Function &F) {
  for (Instruction &I : instructions(F)) {
    if (auto *Comparison = dyn_cast<ICmpInst>(&I))
      WorkList.push_back(Comparison);
  }

  bool Changed = false;
  while (not WorkList.empty()) {
    ICmpInst *Comparison = WorkList.pop_back_val();
    // Assume canonical operand order: constants are on the right.
    if (not Comparison->isEquality()
        or not match(Comparison->getOperand(1), m_ZeroInt())) {
      continue;
    }

    // Distribution erases Comparison and queues its new comparisons.
    if (distributeOrComparison(*Comparison)) {
      Changed = true;
      continue;
    }

    // Always try both rewrites, even if an earlier comparison changed.
    Changed = canonicalizeShiftedMask(*Comparison) or Changed;
    Changed = simplifySignBitComparison(*Comparison) or Changed;
  }

  return Changed;
}

} // namespace

char SimplifyComparisonsWrapperPass::ID = 0;
using Register = RegisterPass<SimplifyComparisonsWrapperPass>;
static Register
  X("simplify-comparisons", "Simplify integer comparisons", false, false);

bool SimplifyComparisonsWrapperPass::runOnFunction(Function &F) {
  return ComparisonSimplifier().run(F);
}

PreservedAnalyses
revng::SimplifyComparisonsPass::run(Function &F, FunctionAnalysisManager &) {
  return ComparisonSimplifier().run(F) ? PreservedAnalyses::none() :
                                         PreservedAnalyses::all();
}
