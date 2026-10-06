/// A set of simple transformation that shrink the type of certain instructions.
/// This should be run before TypeShrinking.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Support/IRBuilder.h"

using namespace llvm;
using std::min;

namespace {

/// The bits of Source a rebuild request asks for: the NarrowType-wide window
/// starting at bit Exponent, that is `trunc(lshr(Source, Exponent))`.
///
/// This is what identifies a rebuilt narrow value. Source alone does not: a
/// single traversal asks for several windows of the same value, because `mul`
/// splits the exponent between its operands and `shl` re-anchors it.
struct BitWindow {
  llvm::Value *Source = nullptr;
  unsigned Exponent = 0;
  llvm::Type *NarrowType = nullptr;

  bool operator==(const BitWindow &) const = default;
};

} // namespace

namespace llvm {

template<>
struct DenseMapInfo<BitWindow> {
  /// Real windows always name a value, so reusing the reserved pointers of
  /// Value * keeps the special keys distinct from every window.
  static BitWindow getEmptyKey() {
    return { DenseMapInfo<Value *>::getEmptyKey(), 0, nullptr };
  }

  static BitWindow getTombstoneKey() {
    return { DenseMapInfo<Value *>::getTombstoneKey(), 0, nullptr };
  }

  static unsigned getHashValue(const BitWindow &Window) {
    return static_cast<unsigned>(hash_combine(Window.Source,
                                              Window.Exponent,
                                              Window.NarrowType));
  }

  static bool isEqual(const BitWindow &LHS, const BitWindow &RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm

namespace {

class EarlyTypeShrinking : public llvm::FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  EarlyTypeShrinking() : FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {}
};

char EarlyTypeShrinking::ID = 0;

/// Find the narrowest emittable type after removing a power-of-two factor.
/// FactorExponent bounds the number of low zero bits that can be removed.
static Type *getNarrowType(Type *OuterType, unsigned FactorExponent) {
  unsigned Width = OuterType->getIntegerBitWidth();
  for (unsigned NarrowWidth : { 8U, 16U, 32U, 64U }) {
    if (NarrowWidth < Width and Width - NarrowWidth <= FactorExponent)
      return IntegerType::get(OuterType->getContext(), NarrowWidth);
  }
  return nullptr;
}

/// Identify removable powers of two in expressions and rebuild their quotients.
///
/// An expression has the form W * 2^A, modulo its integer width. For addition,
/// subtraction and bitwise operations, the goal is to identify a power-of-two
/// factor common to both operands and remove it from each before rebuilding.
/// Multiplication instead combines the factors of its operands. Constants and
/// left shifts supply factors; explicit shifts and constant operands are not
/// required in the operations that propagate them.
///
/// Unlike known-bits analysis, this only recognizes factors we can remove by
/// rebuilding arithmetic, without introducing wide right shifts.
///
/// For example, both addends have a factor of 2^24, so
/// getFactorExponent(%sum) returns 24:
///
///     %shifted = shl i32 %x, 24
///     %sum = add i32 %shifted, 16777216 ; 1 << 24
///
/// Rebuilding %sum with Exponent = 24 and NarrowType = i8 produces:
///
///     %low = trunc i32 %x to i8
///     %narrow = add i8 %low, 1
///
/// The consuming pass can then replace `ashr i32 %sum, 24` with
/// `sext i8 %narrow to i32`.
///
/// A multiply can supply the same factor without an explicit left shift:
///
///     %product = mul i32 %x, 83886080 ; 5 << 24
///
/// getFactorExponent(%product) returns 24, and rebuilding it in i8 produces:
///
///     %low = trunc i32 %x to i8
///     %narrow = mul i8 %low, 5
///
/// Factors can also come from both operands and propagate through bitwise
/// operations. getFactorExponent(%bits) returns 24 for this expression tree:
///
///     %a = shl i32 %x, 12
///     %b = shl i32 %y, 12
///     %product = mul i32 %a, %b
///     %c = shl i32 %z, 24
///     %bits = xor i32 %product, %c
///
/// Rebuilding %bits with Exponent = 24 and NarrowType = i8 produces:
///
///     %low.x = trunc i32 %x to i8
///     %low.y = trunc i32 %y to i8
///     %product.narrow = mul i8 %low.x, %low.y
///     %low.z = trunc i32 %z to i8
///     %narrow = xor i8 %product.narrow, %low.z
///
/// getFactorExponent inspects existing IR and caches the factors it recognizes.
/// rebuild inserts the corresponding narrow computation at the builder's
/// insertion point. The caller replaces the original consumer with that result
/// and any necessary casts or shifts.
class PowerOfTwoFactorization {
private:
  /// Binary instructions the analysis visits before treating the rest of an
  /// expression as opaque. Bounds the cost of a single root; raising it only
  /// widens the expressions the pass is willing to look at.
  static constexpr unsigned InstructionBudget = 64;

private:
  /// Exponent recognized for each analyzed value. Pure analysis over IR this
  /// instance does not modify, so entries stay valid as long as that IR does,
  /// and could be shared across consumers.
  llvm::DenseMap<Value *, unsigned> FactorExponents;

  /// Narrow values already materialized, by the window they rebuild. Unlike
  /// the analysis cache this holds *new* instructions, all emitted at one
  /// insertion point, so it must not be reused for a consumer they would not
  /// dominate.
  llvm::DenseMap<BitWindow, Value *> Rebuilt;

  /// Budget left for the root currently being analyzed. Reset by
  /// getFactorExponent, decremented by computeFactorExponent.
  unsigned RemainingInstructions = InstructionBudget;

public:
  /// Return the exponent of a power-of-two factor that rebuild can remove
  /// from V, analyzing it as a new root with a fresh traversal budget.
  ///
  /// Call this once per expression to factor. Values analyzed for an earlier
  /// root keep their cached exponents and cost nothing, so the budget bounds
  /// the work per root rather than per instance: the two operands of one
  /// comparison each get a full budget and neither can starve the other.
  unsigned getFactorExponent(Value *V) {
    RemainingInstructions = InstructionBudget;
    return computeFactorExponent(V);
  }

private:
  /// Recursive worker behind getFactorExponent, sharing one root's budget.
  ///
  /// For a scalar integer V, the result A means that every defined value of V
  /// is a multiple of 2^A, with at least A low zero bits. A is between zero and
  /// V's bit width and may be smaller than the largest possible exponent.
  /// Unsupported expressions and expressions beyond the traversal budget
  /// contribute zero, which keeps the result a valid lower bound: running out
  /// of budget narrows less, it never reports a factor that is not there.
  ///
  /// For example, identify the factor common to the two addends:
  ///
  ///     %shifted = shl i32 %x, 24
  ///     %sum = add i32 %shifted, 65536 ; 1 << 16
  ///
  /// The operands have factors of 2^24 and 2^16. Their common factor is 2^16,
  /// so getFactorExponent(%sum) returns min(24, 16) = 16. For multiplication,
  /// the exponents would instead add, capped at the integer width.
  ///
  /// This method updates its analysis cache but does not insert, modify or
  /// erase LLVM instructions. Keep the analyzed expression unchanged while
  /// using this instance to rebuild it.
  RecursiveCoroutine<unsigned> computeFactorExponent(Value *V) {
    using namespace PatternMatch;

    revng_assert(V->getType()->isIntegerTy());

    if (auto *C = dyn_cast<ConstantInt>(V))
      rc_return C->getValue().countTrailingZeros();

    auto [It, Inserted] = FactorExponents.try_emplace(V, 0);
    if (not Inserted)
      rc_return It->second;

    auto *Binary = dyn_cast<BinaryOperator>(V);
    if (Binary == nullptr or RemainingInstructions == 0)
      rc_return 0;
    --RemainingInstructions;

    unsigned Width = V->getType()->getIntegerBitWidth();
    unsigned Exponent = 0;
    Value *Operand = nullptr;
    const APInt *Shift = nullptr;
    if (match(V, m_Shl(m_Value(Operand), m_APInt(Shift)))) {
      // Invalid and variable shifts remain opaque.
      if (Shift->ult(Width)) {
        unsigned OperandExponent = rc_recur computeFactorExponent(Operand);
        Exponent = min(Width,
                       unsigned(Shift->getZExtValue()) + OperandExponent);
      }
    } else {
      switch (Binary->getOpcode()) {
      case Instruction::Add:
      case Instruction::Sub:
      case Instruction::And:
      case Instruction::Or:
      case Instruction::Xor:
      case Instruction::Mul: {
        Value *LHS = Binary->getOperand(0);
        Value *RHS = Binary->getOperand(1);
        unsigned LHSExponent = rc_recur computeFactorExponent(LHS);
        unsigned RHSExponent = rc_recur computeFactorExponent(RHS);
        // Multiply combines factors; otherwise take the common factor
        Exponent = Binary->getOpcode() == Instruction::Mul ?
                     min(Width, LHSExponent + RHSExponent) :
                     min(LHSExponent, RHSExponent);
        break;
      }
      default:
        break;
      }
    }

    // Visiting operands may have invalidated It by inserting cache entries.
    FactorExponents[V] = Exponent;
    rc_return Exponent;
  }

public:
  /// Insert a narrow computation of V with the factor 2^Exponent removed.
  ///
  /// For defined V, compute the unsigned quotient V / 2^Exponent in NarrowType.
  /// Require Exponent to be at most getFactorExponent(V), and NarrowType to be
  /// narrower than V and fit in the remaining bits. Exponent may be smaller
  /// than the recognized exponent.
  ///
  /// Written out, the result is `trunc(lshr(V, Exponent), NarrowType)`, the
  /// BitWindow below. Each case preserves that equation while turning the
  /// request into requests for windows of V's operands, which is what lets the
  /// quotient be computed without ever emitting the wide right shift the
  /// equation names.
  ///
  /// For example, given:
  ///
  ///     %shifted = shl i32 %x, 24
  ///     %sum = add i32 %shifted, 16777216 ; 1 << 24
  ///
  /// rebuild(B, %sum, 24, i8) inserts:
  ///
  ///     %low = trunc i32 %x to i8
  ///     %narrow = add i8 %low, 1
  ///
  /// It returns %narrow, which represents the high eight bits of %sum.
  /// The caller decides how to extend or shift this result for its consumer.
  /// Original instructions and their uses are left intact. New instructions
  /// are inserted at B's current insertion point, without overflow flags, and
  /// cached so shared subexpressions are only rebuilt once. With Exponent zero,
  /// simply truncate V instead of visiting its operands.
  RecursiveCoroutine<Value *>
  rebuild(revng::IRBuilder &B, Value *V, unsigned Exponent, Type *NarrowType) {
    BitWindow Window{ V, Exponent, NarrowType };
    if (auto It = Rebuilt.find(Window); It != Rebuilt.end())
      rc_return It->second;

    // This only feeds the assertion: rebuild never widens the exponent it was
    // given. Nor can it consume budget, because every node it reaches is
    // already cached. Descending below a node requires that node's exponent to
    // be nonzero, which is exactly the case in which the analysis recursed
    // into its operands; a zero exponent truncates V and stops here. Adding an
    // opcode that reports a factor without visiting its operands would break
    // that, and the assertion would catch it.
    unsigned AvailableExponent = rc_recur computeFactorExponent(V);
    revng_assert(Exponent <= AvailableExponent);
    unsigned Width = NarrowType->getIntegerBitWidth();
    revng_assert(Width + Exponent <= V->getType()->getIntegerBitWidth());

    Value *Result = nullptr;
    if (Exponent == 0) {
      Result = B.CreateTrunc(V, NarrowType);
    } else if (auto *C = dyn_cast<ConstantInt>(V)) {
      Result = ConstantInt::get(NarrowType,
                                C->getValue().lshr(Exponent).trunc(Width));
    } else {
      auto *Binary = cast<BinaryOperator>(V);
      Value *LHS = Binary->getOperand(0);
      Value *RHS = Binary->getOperand(1);
      if (Binary->getOpcode() == Instruction::Shl) {
        unsigned Shift = cast<ConstantInt>(RHS)->getZExtValue();
        if (Shift < Exponent) {
          Result = rc_recur rebuild(B, LHS, Exponent - Shift, NarrowType);
        } else if (Shift - Exponent >= Width) {
          // Truncation discards every bit. Do not create a narrow overshift.
          Result = ConstantInt::get(NarrowType, 0);
        } else {
          Result = rc_recur rebuild(B, LHS, 0, NarrowType);
          if (Shift != Exponent)
            Result = B.CreateShl(Result, Shift - Exponent);
        }
      } else {
        unsigned LHSExponent = Exponent;
        unsigned RHSExponent = Exponent;
        if (Binary->getOpcode() == Instruction::Mul) {
          // Split the factor across the two multiplicands
          unsigned AvailableLHSExponent = rc_recur computeFactorExponent(LHS);
          LHSExponent = min(Exponent, AvailableLHSExponent);
          RHSExponent = Exponent - LHSExponent;
        }

        Value *NarrowLHS = rc_recur rebuild(B, LHS, LHSExponent, NarrowType);
        Value *NarrowRHS = rc_recur rebuild(B, RHS, RHSExponent, NarrowType);
        // Do not transfer nowrap/exact flags to narrower arithmetic.
        Result = B.CreateBinOp(Binary->getOpcode(), NarrowLHS, NarrowRHS);
      }
    }

    Rebuilt[Window] = Result;
    rc_return Result;
  }
};

/// Narrow a right shift's input by removing a power-of-two factor.
///
/// Identify a removable factor 2^A in the input V, which may be an arithmetic
/// expression rather than a left shift. Removing it leaves an OuterSize - A
/// bit quotient T. Choose A so this width is emittable, then rebuild T in that
/// type. A can be smaller than the exponent identified by the analysis.
///
/// For `V >> B`, adjust the shift by the removed exponent A and extend T back
/// to OuterSize. `ashr` copies V's top bit, which is T's top bit, so it needs
/// sign extension; `lshr` needs zero extension.
///
/// An explicit left shift is the simplest source of a factor, with T obtained
/// by truncating its input:
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
/// In the last case, extension fills the A bits above T with copies of its
/// sign bit or zero. Shifting left by A - B only discards some of those copies.
static bool shrinkRightShift(revng::IRBuilder &B, Instruction &I) {
  using namespace PatternMatch;

  const APInt *MatchedAmount = nullptr;
  Value *Input = nullptr;

  if (not I.getType()->isIntegerTy()
      or not match(&I, m_Shr(m_Value(Input), m_APInt(MatchedAmount)))) {
    return false;
  }

  Type *OuterType = I.getType();
  uint64_t OuterSize = OuterType->getIntegerBitWidth();

  if (MatchedAmount->uge(OuterSize))
    return false;

  PowerOfTwoFactorization Factors;
  unsigned FactorExponent = Factors.getFactorExponent(Input);
  Type *InnerType = getNarrowType(OuterType, FactorExponent);
  if (InnerType == nullptr)
    return false;

  uint64_t RemovedExponent = OuterSize - InnerType->getIntegerBitWidth();
  uint64_t RightShiftAmount = MatchedAmount->getZExtValue();

  B.SetInsertPoint(&I, I.getDebugLoc());
  Value *NarrowValue = Factors.rebuild(B, Input, RemovedExponent, InnerType);

  const bool IsArithmetic = I.isArithmeticShift();

  if (RightShiftAmount > RemovedExponent) {
    uint64_t InnerRightShift = RightShiftAmount - RemovedExponent;
    NarrowValue = IsArithmetic ? B.CreateAShr(NarrowValue, InnerRightShift) :
                                 B.CreateLShr(NarrowValue, InnerRightShift);
  }

  Value *Replacement = IsArithmetic ? B.CreateSExt(NarrowValue, OuterType) :
                                      B.CreateZExt(NarrowValue, OuterType);

  if (RightShiftAmount < RemovedExponent) {
    Replacement = B.CreateShl(Replacement, RemovedExponent - RightShiftAmount);
  }

  I.replaceAllUsesWith(Replacement);
  I.eraseFromParent();
  return true;
}

/// Remove a common power-of-two factor from both comparison operands, using
/// the same decomposition as shrinkRightShift.
///
/// The goal is to identify a factor 2^K common to both operands, taking the
/// minimum of their recognized exponents. Choose K so OuterSize - K is an
/// emittable width, then rebuild both quotients at that width and compare them.
/// The operands need not contain explicit shifts or have equal exponents.
///
/// For example, both operands below have the common factor 2^32:
///
///     %lhs = mul i64 %x, 12884901888 ; 3 * 2^32
///     %rhs = mul i64 %y, 21474836480 ; 5 * 2^32
///     %cmp = icmp slt i64 %lhs, %rhs
///
/// Removing it produces:
///
///     %low.x = trunc i64 %x to i32
///     %lhs.narrow = mul i32 %low.x, 3
///     %low.y = trunc i64 %y to i32
///     %rhs.narrow = mul i32 %low.y, 5
///     %cmp = icmp slt i32 %lhs.narrow, %rhs.narrow
///
/// Every predicate survives: each original operand is its narrow quotient,
/// sign- or zero-extended, times 2^K without overflow at the original width.
/// This common positive multiplier preserves signed order, unsigned order and
/// equality.
static bool shrinkComparison(revng::IRBuilder &B, Instruction &I) {
  auto *Compare = dyn_cast<ICmpInst>(&I);
  if (Compare == nullptr)
    return false;

  Value *LHS = Compare->getOperand(0);
  Value *RHS = Compare->getOperand(1);

  Type *OperandType = LHS->getType();
  if (not OperandType->isIntegerTy())
    return false;

  PowerOfTwoFactorization Factors;
  unsigned LHSExponent = Factors.getFactorExponent(LHS);
  unsigned RHSExponent = Factors.getFactorExponent(RHS);
  unsigned CommonExponent = min(LHSExponent, RHSExponent);
  Type *InnerType = getNarrowType(OperandType, CommonExponent);
  if (InnerType == nullptr)
    return false;

  unsigned RemovedExponent = OperandType->getIntegerBitWidth()
                             - InnerType->getIntegerBitWidth();

  B.SetInsertPoint(&I, I.getDebugLoc());
  Value *NarrowLHS = Factors.rebuild(B, LHS, RemovedExponent, InnerType);
  Value *NarrowRHS = Factors.rebuild(B, RHS, RemovedExponent, InnerType);
  Value *Replacement = B.CreateICmp(Compare->getPredicate(),
                                    NarrowLHS,
                                    NarrowRHS);

  I.replaceAllUsesWith(Replacement);
  I.eraseFromParent();
  return true;
}

bool EarlyTypeShrinking::runOnFunction(Function &F) {
  bool Changed = false;

  // TODO: checks are only omitted here because of unit tests.
  revng::IRBuilder B(F.getContext());

  for (Instruction &I : llvm::make_early_inc_range(llvm::instructions(F))) {
    if (shrinkRightShift(B, I) or shrinkComparison(B, I))
      Changed = true;
  }

  return Changed;
}

} // namespace

static RegisterPass<EarlyTypeShrinking> Y("early-type-shrinking",
                                          "Preliminary instruction type "
                                          "shrinking",
                                          true,
                                          true);
