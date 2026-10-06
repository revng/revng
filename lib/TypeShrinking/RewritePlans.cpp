/// Select narrower representations of integer computations, and rebuild
/// them accordingly.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"

#include "revng/Support/CommandLine.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRBuilder.h"
#include "revng/TypeShrinking/RewritePlans.h"

#include "SCCPValueRanges.h"

using namespace llvm;
using std::max;
using std::min;

static cl::opt<uint32_t> MinimumWidth("min-width",
                                      cl::init(8),
                                      cl::desc("Minimum emitted integer width"),
                                      cl::value_desc("min-width"),
                                      cl::cat(MainCategory));

namespace TypeShrinking {

/// Return the integer width, or zero for a non-integer value.
static unsigned width(const Value *V) {
  return V->getType()->isIntegerTy() ? V->getType()->getIntegerBitWidth() : 0;
}

/// Whether the opcode is integer division or remainder.
static bool isDivision(unsigned Opcode) {
  return Opcode == Instruction::UDiv or Opcode == Instruction::SDiv
         or Opcode == Instruction::URem or Opcode == Instruction::SRem;
}

static unsigned roundWidth(unsigned Required, unsigned Original) {
  Required = max(Required, MinimumWidth.getValue());

  for (unsigned Candidate : { 8U, 16U, 32U, 64U }) {
    if (Candidate >= Required and Candidate < Original)
      return Candidate;
  }
  return Original;
}

RewritePlan RewritePlan::create(Instruction &I,
                                unsigned Demand,
                                const SCCPValueRanges &Ranges) {
  // Comparisons preserve the operands' values, rather than their low bits.
  if (auto *Compare = dyn_cast<ICmpInst>(&I)) {
    unsigned OperandWidth = width(Compare->getOperand(0));
    if (OperandWidth == 0)
      return RewritePlan();
    ConstantRange A = Ranges.range(Compare->getOperand(0));
    ConstantRange B = Ranges.range(Compare->getOperand(1));
    unsigned U = max(A.getActiveBits(), B.getActiveBits());
    unsigned S = max(A.getMinSignedBits(), B.getMinSignedBits());
    U = roundWidth(U, OperandWidth);
    S = roundWidth(S, OperandWidth);

    // A narrower unsigned representation makes both operands nonnegative
    // at their original width, so unsigned predicates preserve their order.
    auto Predicate = Compare->getPredicate();
    if (U <= S and U < OperandWidth)
      Predicate = Compare->getUnsignedPredicate();

    return RewritePlan(ComparisonRewrite{ min(U, S), Predicate });
  }

  unsigned W = width(&I);
  if (W == 0)
    return RewritePlan();

  // Only plan operations supported by the rebuilder.
  switch (I.getOpcode()) {
  case Instruction::PHI:
  case Instruction::Select:
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::UDiv:
  case Instruction::URem:
  case Instruction::SDiv:
  case Instruction::SRem:
    break;
  default:
    return RewritePlan();
  }

  // Keep the cheaper zero/sign representation of the demanded result bits.
  ConstantRange Bounds = Ranges.range(&I);
  unsigned U = roundWidth(min(Demand, Bounds.getActiveBits()), W);
  unsigned S = roundWidth(min(Demand, Bounds.getMinSignedBits()), W);
  unsigned ResultWidth = min(U, S);
  bool SignExtend = S < U;

  // Extending the result back reproduces the original value when its bounds,
  // rather than only its demand, fit the result.
  unsigned ValueWidth = SignExtend ? Bounds.getMinSignedBits() :
                                     Bounds.getActiveBits();
  IntegerRewrite P{ ResultWidth,
                    ResultWidth,
                    SignExtend ? ExtensionKind::Sign : ExtensionKind::Zero,
                    ValueWidth <= ResultWidth };
  unsigned Required = min(Demand, P.ResultWidth);

  // Shift execution must retain the source bits and exceed every shift amount.
  if (I.isShift()) {
    ConstantRange Amounts = Ranges.range(I.getOperand(1));
    unsigned Maximum = Amounts.getUnsignedMax().getLimitedValue(W);
    if (Maximum >= W) {
      // The range does not prove that narrowing keeps the shift defined.
      P.ComputationWidth = W;
    } else {
      unsigned Input = Required;
      if (I.getOpcode() != Instruction::Shl) {
        // Right shifts pull in high bits, capped by the exact input width.
        ConstantRange InputBounds = Ranges.range(I.getOperand(0));
        unsigned Exact = I.getOpcode() == Instruction::LShr ?
                           InputBounds.getActiveBits() :
                           InputBounds.getMinSignedBits();
        Input = min(Required + Maximum, Exact);
      }
      unsigned Execution = max({ P.ResultWidth, Input, Maximum + 1 });
      P.ComputationWidth = roundWidth(Execution, W);
    }
  } else if (isDivision(I.getOpcode())) {
    // Division and remainder need exact operands even for a partial result.
    bool Signed = I.getOpcode() == Instruction::SDiv
                  or I.getOpcode() == Instruction::SRem;
    ConstantRange Left = Ranges.range(I.getOperand(0));
    ConstantRange Right = Ranges.range(I.getOperand(1));
    unsigned Input = Signed ?
                       max(Left.getMinSignedBits(), Right.getMinSignedBits()) :
                       max(Left.getActiveBits(), Right.getActiveBits());
    P.ComputationWidth = roundWidth(max(P.ResultWidth, Input), W);

    // Narrowing must not introduce signed-minimum / -1 overflow.
    if (Signed) {
      while (P.ComputationWidth < W) {
        APInt Minimum = APInt::getSignedMinValue(P.ComputationWidth).sext(W);
        bool CanOverflow = Left.contains(Minimum)
                           and Right.contains(APInt::getAllOnes(W));
        if (not CanOverflow)
          break;

        P.ComputationWidth = roundWidth(P.ComputationWidth + 1, W);
      }
    }
  }
  return RewritePlan(P);
}

void RewritePlans::insert(Instruction &I, const RewritePlan &Plan) {
  revng_assert(Plan);
  bool Inserted = Plans.insert({ &I, Plan }).second;
  revng_assert(Inserted);

  NarrowsAnything |= Plan.getResultWidth() < width(&I);
  if (auto *Compare = Plan.getComparisonRewrite())
    NarrowsAnything |= Compare->OperandWidth < width(I.getOperand(0));
}

bool RewritePlans::apply() {
  if (not NarrowsAnything)
    return false;

  // First, build a replacement for each planned instruction, right before it:
  // the same operation, performed at the widths its plan selects, on operands
  // converted to those widths.
  //
  // Since plans are in reverse post-order, the planned operands of an
  // instruction other than a PHI already have a replacement.
  revng_assert(Replacements.empty());
  for (const auto &[Original, Plan] : Plans)
    Replacements[Original] = rebuild(*Original, Plan);

  // Then, now that every replacement exists, add the incoming values of the
  // new PHIs, each adapted at the end of the predecessor it arrives from.
  finalizePhis();

  // Then, make the instructions without a plan use the replacements. This is
  // not a replaceAllUsesWith, because a replacement can be narrower than its
  // original.
  replaceUses();

  // Finally, the original instructions are only used by each other, and can
  // be erased.
  eraseOriginals();

  return true;
}

Value *RewritePlans::rebuild(Instruction &I, const RewritePlan &Plan) {
  revng::IRBuilder Builder(&I, I.getDebugLoc());
  unsigned ResultWidth = Plan.getResultWidth();
  auto *ResultType = Builder.getIntNTy(ResultWidth);

  Value *Result = nullptr;
  if (auto *Compare = Plan.getComparisonRewrite()) {
    Value *LHS = adapt(Builder, I.getOperand(0), Compare->OperandWidth);
    Value *RHS = adapt(Builder, I.getOperand(1), Compare->OperandWidth);
    Result = Builder.CreateICmp(Compare->Predicate, LHS, RHS);
  } else if (auto *Phi = dyn_cast<PHINode>(&I)) {
    Result = Builder.CreatePHI(ResultType, Phi->getNumIncomingValues());
  } else if (auto *Select = dyn_cast<SelectInst>(&I)) {
    Value *Condition = adapt(Builder, Select->getCondition(), 1);
    Value *True = adapt(Builder, Select->getTrueValue(), ResultWidth);
    Value *False = adapt(Builder, Select->getFalseValue(), ResultWidth);
    Result = Builder.CreateSelect(Condition, True, False);
  } else if (auto *Cast = dyn_cast<CastInst>(&I)) {
    unsigned SourceWidth = Cast->getSrcTy()->getIntegerBitWidth();
    unsigned Width = min(SourceWidth, ResultWidth);
    Value *Operand = adapt(Builder, Cast->getOperand(0), Width);
    Result = Builder.CreateIntCast(Operand, ResultType, isa<SExtInst>(Cast));
  } else {
    unsigned Width = Plan.getIntegerRewrite()->ComputationWidth;
    Value *LHS = adapt(Builder, I.getOperand(0), Width);
    Value *RHS = adapt(Builder, I.getOperand(1), Width);
    // Fresh operations do not inherit nowrap/exact flags, which narrowing
    // can invalidate even when the original operation had those flags.
    auto Opcode = Instruction::BinaryOps(I.getOpcode());
    Value *Operation = Builder.CreateBinOp(Opcode, LHS, RHS);
    Result = Builder.CreateTrunc(Operation, ResultType);
  }

  return Result;
}

void RewritePlans::finalizePhis() {
  for (const auto &[Original, Plan] : Plans) {
    auto *Phi = dyn_cast<PHINode>(Original);
    if (Phi == nullptr)
      continue;

    auto *NewPhi = cast<PHINode>(Replacements.lookup(Phi));
    unsigned Width = Plan.getResultWidth();
    for (unsigned Index = 0; Index < Phi->getNumIncomingValues(); ++Index) {
      BasicBlock *Predecessor = Phi->getIncomingBlock(Index);
      Value *Incoming = Phi->getIncomingValue(Index);
      const DebugLoc &Location = Phi->getDebugLoc();
      Value *Adapted = adaptOnEdge(Predecessor, Incoming, Width, Location);
      NewPhi->addIncoming(Adapted, Predecessor);
    }
  }
}

void RewritePlans::replaceUses() {
  for (const auto &[Original, Plan] : Plans) {
    unsigned Width = Original->getType()->getIntegerBitWidth();
    const DebugLoc &Location = Original->getDebugLoc();
    for (Use &Operand : make_early_inc_range(Original->uses())) {
      auto *User = cast<Instruction>(Operand.getUser());
      if (find(User) != nullptr)
        continue;

      if (auto *Phi = dyn_cast<PHINode>(User)) {
        BasicBlock *Predecessor = Phi->getIncomingBlock(Operand);
        Operand.set(adaptOnEdge(Predecessor, Original, Width, Location));
      } else {
        revng::IRBuilder Builder(User, Location);
        Operand.set(adapt(Builder, Original, Width));
      }

      // Demand analysis ignores flags, make sure we don't have them, unless
      // the replacement reproduces the whole original value.
      bool IsInteger = User->getType()->isIntegerTy();
      bool HasFlags = IsInteger and User->hasPoisonGeneratingFlags();
      revng_assert(Plan.isLossless() or not HasFlags);
    }
  }
}

void RewritePlans::eraseOriginals() {
  for (Instruction *Original : make_first_range(Plans))
    Original->dropAllReferences();

  for (Instruction *Original : make_first_range(Plans)) {
    revng_assert(Original->use_empty());
    Original->eraseFromParent();
  }
}

/// Convert \p V to \p Width at the insertion point of \p Builder. A planned
/// instruction is represented by its replacement, extended as its plan selects.
Value *
RewritePlans::adapt(revng::IRBuilder &Builder, Value *V, unsigned Width) {
  bool Signed = false;
  if (auto *I = dyn_cast<Instruction>(V)) {
    if (const RewritePlan *Plan = find(I)) {
      // Planning in reverse post-order rebuilds every operand before the
      // instruction using it, and PHIs are only completed at the end.
      V = Replacements.lookup(I);
      revng_assert(V != nullptr);

      if (auto *Integer = Plan->getIntegerRewrite())
        Signed = Integer->ResultExtension == ExtensionKind::Sign;
    }
  }
  return Builder.CreateIntCast(V, Builder.getIntNTy(Width), Signed);
}

/// Adapt \p V at the end of \p Predecessor, where its edge to a PHI is taken.
/// The result is shared, because LLVM requires a PHI to receive the same value
/// on every edge from the same predecessor.
Value *RewritePlans::adaptOnEdge(BasicBlock *Predecessor,
                                 Value *V,
                                 unsigned Width,
                                 const DebugLoc &Location) {
  auto [It, Inserted] = EdgeValues.try_emplace({ Predecessor, V, Width });
  if (Inserted) {
    revng::IRBuilder Builder(Predecessor->getTerminator(), Location);
    It->second = adapt(Builder, V, Width);
  }
  return It->second;
}

class TypeShrinkingAnnotatedWriter : public AssemblyAnnotationWriter {
private:
  const RewritePlans &Plans;

public:
  TypeShrinkingAnnotatedWriter(const RewritePlans &Plans) : Plans(Plans) {}

public:
  void emitInstructionAnnot(const Instruction *I,
                            formatted_raw_ostream &Stream) final {
    const RewritePlan *Plan = Plans.find(const_cast<Instruction *>(I));
    if (Plan == nullptr)
      return;

    if (auto *Integer = Plan->getIntegerRewrite()) {
      bool Signed = Integer->ResultExtension == ExtensionKind::Sign;
      StringRef Extension = Signed ? "sign" : "zero";
      Stream << "  ; Result width: " << Integer->ResultWidth << "\n";
      Stream << "  ; Computation width: " << Integer->ComputationWidth << "\n";
      Stream << "  ; Result extension: " << Extension << "\n";
    } else {
      const auto &Compare = *Plan->getComparisonRewrite();
      Stream << "  ; Operand width: " << Compare.OperandWidth << "\n";
      Stream << "  ; Predicate: "
             << CmpInst::getPredicateName(Compare.Predicate) << "\n";
    }
  }
};

void RewritePlans::dump() const {
  TypeShrinkingAnnotatedWriter Annotator(*this);
  raw_os_ostream Stream(dbg);
  F.print(Stream, &Annotator);
}

} // namespace TypeShrinking
