/// \file DataFlowRangeAnalysis.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"

#include "revng/ADT/ConstantRangeSet.h"
#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/ValueMaterializer/DataFlowRangeAnalysis.h"

static Logger Log("data-flow-range-analysis");

static bool isSigned(llvm::ICmpInst::Predicate Predicate) {
  switch (Predicate) {
  case llvm::CmpInst::ICMP_SLE:
  case llvm::CmpInst::ICMP_SGT:
  case llvm::CmpInst::ICMP_SGE:
  case llvm::CmpInst::ICMP_SLT:
    return true;

  case llvm::CmpInst::ICMP_UGE:
  case llvm::CmpInst::ICMP_ULT:
  case llvm::CmpInst::ICMP_ULE:
  case llvm::CmpInst::ICMP_UGT:
    return false;

  default:
    revng_abort();
  }
}

static bool isInclusive(llvm::ICmpInst::Predicate Predicate) {
  switch (Predicate) {
  case llvm::CmpInst::ICMP_SLE:
  case llvm::CmpInst::ICMP_ULE:
  case llvm::CmpInst::ICMP_SGT:
  case llvm::CmpInst::ICMP_UGT:
    return true;

  case llvm::CmpInst::ICMP_SGE:
  case llvm::CmpInst::ICMP_UGE:
  case llvm::CmpInst::ICMP_SLT:
  case llvm::CmpInst::ICMP_ULT:
    return false;

  default:
    revng_abort();
  }
}

static bool isLowerThan(llvm::ICmpInst::Predicate Predicate) {
  switch (Predicate) {
  case llvm::CmpInst::ICMP_SLE:
  case llvm::CmpInst::ICMP_ULE:
  case llvm::CmpInst::ICMP_SLT:
  case llvm::CmpInst::ICMP_ULT:
    return true;

  case llvm::CmpInst::ICMP_SGE:
  case llvm::CmpInst::ICMP_UGE:
  case llvm::CmpInst::ICMP_SGT:
  case llvm::CmpInst::ICMP_UGT:
    return false;

  default:
    revng_abort();
  }
}

// x + c1 CMP_?? c2
static ConstantRangeSet
getRangeFromInequality(const llvm::APInt &C1,
                       llvm::ICmpInst::Predicate Predicate,
                       const llvm::APInt &C2) {
  using namespace llvm;
  auto BitWidth = C1.getBitWidth();
  revng_assert(C2.getBitWidth() == BitWidth);
  bool Signed = isSigned(Predicate);

  // Determine min and max
  APInt Min;
  APInt Max;
  if (Signed) {
    Min = APInt::getSignedMinValue(BitWidth);
    Max = APInt::getSignedMaxValue(BitWidth);
  } else {
    Min = APInt::getMinValue(BitWidth);
    Max = APInt::getMaxValue(BitWidth);
  }

  // Determine start and stop ranges
  auto Start = APInt::getMinValue(BitWidth) - C1;
  auto Stop = C2 - C1;

  // Add +1 to stop if we're dealing with <= or >
  if (isInclusive(Predicate))
    Stop += 1;

  // Check if we can represent this with a single range or if we need to
  // consider wrap-around
  bool Ordered = false;
  if (Signed)
    Ordered = Start.slt(Stop);
  else
    Ordered = Start.ult(Stop);

  ConstantRangeSet Result;
  if (Start == Stop) {
    Result = ConstantRange::getEmpty(BitWidth);
  } else if (Ordered) {
    Result = ConstantRange(Start, Stop);
  } else {
    Result = ConstantRangeSet({ Start, Max + 1 })
             | ConstantRangeSet({ Min, Stop });
  }

  if (not isLowerThan(Predicate))
    Result = ~Result;

  return Result;
}

namespace revng::detail {

class Visitor {
private:
  using CacheEntry = DataFlowRangeAnalysis::CacheEntry;

private:
  llvm::DenseSet<llvm::Value *> Stack;
  std::map<CacheEntry, ConstantRangeSet> &Cache;
  llvm::Value &Variable;
  llvm::ModuleSlotTracker &MST;

public:
  Visitor(std::map<CacheEntry, ConstantRangeSet> &Cache,
          llvm::Value &Variable,
          llvm::ModuleSlotTracker &MST) :
    Cache(Cache), Variable(Variable), MST(MST) {}

public:
  RecursiveCoroutine<std::optional<ConstantRangeSet>>
  visit(llvm::Value &Constraint);

private:
  ConstantRangeSet *tryGet(llvm::Value &I) {
    auto It = Cache.find({ &I, &Variable });
    if (It == Cache.end())
      return nullptr;
    return &It->second;
  }

  std::optional<ConstantRangeSet>
  record(llvm::Value &I, std::optional<ConstantRangeSet> &&Result) {
    if (not Result.has_value()) {
      revng_log(Log, "Returning an empty result");
      return Result;
    }

    if (Log.isEnabled()) {
      Log << "Returning ";
      Result.value().dump(Log);
      Log << DoLog;
    }

    revng_assert(tryGet(I) == nullptr);
    Cache[{ &I, &Variable }] = Result.value();
    return Result;
  }

  void pop(llvm::Value &I) {
    auto It = Stack.find(&I);
    revng_assert(It != Stack.end());
    Stack.erase(It);
  }

  class StackEntry {
  private:
    Visitor *V = nullptr;
    llvm::Value *I = nullptr;

  public:
    StackEntry(Visitor &V, llvm::Value &I) : V(&V), I(&I) {}

    StackEntry(StackEntry &&Other) {
      V = Other.V;
      I = Other.I;
      Other.I = nullptr;
    }

    StackEntry &operator=(StackEntry &&Other) {
      V = Other.V;
      I = Other.I;
      Other.I = nullptr;
      return *this;
    }

  public:
    ~StackEntry() {
      if (I != nullptr)
        V->pop(*I);
    }
  };

  std::optional<StackEntry> newStackEntry(llvm::Value &I) {
    if (Stack.find(&I) != Stack.end())
      return std::nullopt;
    Stack.insert(&I);
    return StackEntry(*this, I);
  }
};
} // namespace revng::detail

std::optional<ConstantRangeSet>
DataFlowRangeAnalysis::visit(llvm::Value &Constraint, llvm::Value &Variable) {
  revng_log(Log, "New analysis");
  LoggerIndent Indent(Log);

  if (Log.isEnabled()) {
    Log << "Constraint: ";
    Constraint.print(*Log.getAsLLVMStream(), MST, true);
    Log << DoLog;

    Log << "Variable: ";
    Variable.print(*Log.getAsLLVMStream(), MST, true);
    Log << DoLog;
  }

  revng::detail::Visitor V(Cache, Variable, MST);
  return V.visit(Constraint);
}

// TODO: emit graph while performing the visit
// TODO: cache std::nullopt?
inline RecursiveCoroutine<std::optional<ConstantRangeSet>>
revng::detail::Visitor::visit(llvm::Value &Constraint) {
  if (Log.isEnabled()) {
    Log << "Visiting ";
    Constraint.print(*Log.getAsLLVMStream(), MST, true);
    Log << DoLog;
  }
  LoggerIndent Indent(Log);

  // Check cache
  if (auto *Ranges = tryGet(Constraint)) {
    revng_log(Log, "Found in cache: " << Ranges->toString());
    rc_return *Ranges;
  }

  // Start processing current instruction, unless there's recursion
  auto &&MaybeStackEntry = newStackEntry(Constraint);
  bool RecursionDetected = not MaybeStackEntry.has_value();
  if (RecursionDetected) {
    revng_log(Log, "Recursion detected, bailing out");
    rc_return std::nullopt;
  }

  using namespace llvm::PatternMatch;

  // Note: from here on, remember to wrap in record(Constraint, ...) each
  // returned value

  //
  // Handle constants
  //
  if (auto *Constant = dyn_cast<llvm::ConstantInt>(&Constraint)) {
    revng_log(Log, "Handling ConstantInt");
    const llvm::APInt &Value = Constant->getValue();
    auto BitWidth = Variable.getType()->getIntegerBitWidth();
    rc_return record(Constraint, ConstantRangeSet(BitWidth, !Value.isZero()));
  }

  if (isa<llvm::Constant>(Constraint)) {
    revng_log(Log, "Ignoring non-ConstantInt Constant");
    rc_return std::nullopt;
  }

  //
  // Handle constraints on Variable
  //

  // Handle x - 4 < 5 as x >= 4 and x < 9
  using namespace llvm;
  ICmpInst::Predicate Predicate{};
  ConstantInt *Addend = nullptr;
  ConstantInt *Bound = nullptr;
  if (match(&Constraint,
            m_ICmp(Predicate,
                   m_Add(m_Specific(&Variable), m_ConstantInt(Addend)),
                   m_ConstantInt(Bound)))
      and ICmpInst::isRelational(Predicate)) {
    revng_log(Log, "Handling inequality");
    rc_return record(Constraint,
                     getRangeFromInequality(Addend->getValue(),
                                            Predicate,
                                            Bound->getValue()));
  }

  // Handle x - 4, i.e., x - 4 != 0, i.e., x != 4
  if (match(&Constraint, m_Add(m_Specific(&Variable), m_ConstantInt(Addend)))) {
    revng_log(Log, "Handling exact comparison");
    rc_return record(Constraint,
                     ~ConstantRangeSet(ConstantRange(-Addend->getValue())));
  }

  // Handle x & 0b1100 == 0b0100, i.e., 0b0100 <= x <= 0b0111
  llvm::ConstantInt *NegatedMask = nullptr;
  llvm::ConstantInt *FixedConstant = nullptr;
  if (match(&Constraint,
            m_ICmp(Predicate,
                   m_And(m_Specific(&Variable), m_ConstantInt(NegatedMask)),
                   m_ConstantInt(FixedConstant)))) {
    bool IsExactComparison = not ICmpInst::isRelational(Predicate);
    const APInt &FixedValue = FixedConstant->getValue();
    auto Masked = FixedValue & NegatedMask->getValue();
    bool FixedValueIsCompatible = Masked == FixedValue;

    APInt MaskValue = ~NegatedMask->getValue();
    if (IsExactComparison and MaskValue.isMask() and FixedValueIsCompatible) {
      revng_log(Log, "Handling mask");
      revng_assert(Predicate == ICmpInst::Predicate::ICMP_EQ
                   or Predicate == ICmpInst::Predicate::ICMP_NE);

      ConstantRangeSet Result({ FixedValue, FixedValue + MaskValue + 1 });

      if (Predicate == ICmpInst::Predicate::ICMP_NE)
        Result = ~Result;

      rc_return record(Constraint, Result);
    }
  }

  //
  // Handle boolean operations from here on
  //

  // Handle comparison with zero
  Value *Operand = nullptr;
  if (match(&Constraint,
            m_ICmp(Predicate, m_Value(Operand), m_SpecificInt(0)))) {
    if (auto MaybeRange = rc_recur visit(*Operand)) {
      revng_log(Log, "Handling comparison with zero");
      if (Predicate == llvm::CmpInst::ICMP_NE) {
        rc_return record(Constraint, std::move(MaybeRange));
      } else if (Predicate == llvm::CmpInst::ICMP_EQ) {
        rc_return record(Constraint, MaybeRange.value().complement());
      }
    }
  }

  // Handle bitwise operations
  Value *LHS = nullptr;
  Value *RHS = nullptr;
  if (match(&Constraint, m_BinOp(m_Value(LHS), m_Value(RHS)))
      and cast<BinaryOperator>(Constraint).isBitwiseLogicOp()) {
    auto MaybeLHSRange = rc_recur visit(*LHS);
    auto MaybeRHSRange = rc_recur visit(*RHS);
    if (MaybeLHSRange and MaybeRHSRange) {
      revng_log(Log, "Handling bitwise operator");
      switch (cast<BinaryOperator>(Constraint).getOpcode()) {
      case llvm::Instruction::And:
        rc_return record(Constraint, *MaybeLHSRange & *MaybeRHSRange);
      case llvm::Instruction::Or:
        rc_return record(Constraint, *MaybeLHSRange | *MaybeRHSRange);
      default:
        // Do nothing
        break;
      }
    }
  }

  // Handle select
  Value *Condition = nullptr;
  Value *TrueValue = nullptr;
  Value *FalseValue = nullptr;
  if (match(&Constraint,
            m_Select(m_Value(Condition),
                     m_Value(TrueValue),
                     m_Value(FalseValue)))) {
    auto MaybeConditionRange = rc_recur visit(*Condition);
    auto MaybeTrueRange = rc_recur visit(*TrueValue);
    auto MaybeFalseRange = rc_recur visit(*FalseValue);
    if (MaybeConditionRange and MaybeTrueRange and MaybeFalseRange) {
      revng_log(Log, "Handling select");
      rc_return record(Constraint,
                       (*MaybeConditionRange & *MaybeTrueRange)
                         | (~*MaybeConditionRange & *MaybeFalseRange));
    }
  }

  if (&Constraint == &Variable) {
    revng_log(Log, "Returning full set");
    rc_return record(Constraint,
                     ConstantRangeSet(Constraint.getType()
                                        ->getIntegerBitWidth(),
                                      true));
  }

  revng_log(Log, "Can't handle, bailing out");
  rc_return std::nullopt;
}
