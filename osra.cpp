/// \file osra.cpp
/// \brief

// Standard includes
#include <cstdint>
#include <vector>

// LLVM includes
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Pass.h"

// Local includes
#include "datastructures.h"
#include "debug.h"
#include "memoryaccess.h"
#include "revamb.h"
#include "ir-helpers.h"
#include "osra.h"

using namespace llvm;

using Predicate = CmpInst::Predicate;
using OSR = OSRAPass::OSR;
using BoundedValue = OSRAPass::BoundedValue;
using CE = ConstantExpr;
using CI = ConstantInt;
using std::pair;
using std::make_pair;
using std::numeric_limits;

const BoundedValue::MergeType AndMerge = BoundedValue::And;
const BoundedValue::MergeType OrMerge = BoundedValue::Or;

template<typename C>
static auto skip(unsigned ToSkip, C &Container)
                 -> iterator_range<decltype(Container.begin())> {
  auto Begin = std::begin(Container);
  while (ToSkip --> 0)
    Begin++;
  return make_range(Begin, std::end(Container));
}

char OSRAPass::ID = 0;

static RegisterPass<OSRAPass> X("osra", "OSRA Pass", false, false);

Constant *OSR::evaluate(Constant *Value, Type *Int64) const {
  Constant *BaseC = CI::get(Int64, Base, BV->isSigned());
  Constant *FactorC = CI::get(Int64, Factor, BV->isSigned());

  return CE::getAdd(BaseC, CE::getMul(FactorC, Value));
}

static bool isPositive(Constant *C, const DataLayout &DL) {
  auto *Zero = CI::get(C->getType(), 0, true);
  auto *Compare = CE::getCompare(CmpInst::ICMP_SGE, C, Zero);
  return getConstValue(Compare, DL)->getLimitedValue();
}

pair<Constant *, Constant *> OSR::boundaries(Type *Int64,
                                             const DataLayout &DL) const {
  Constant *Min = nullptr;
  Constant *Max = nullptr;
  std::tie(Min, Max) = BV->actualBoundaries(Int64);
  Min = evaluate(Min, Int64);
  Max = evaluate(Max, Int64);

  return { Min, Max };
}

/// \brief Combine two constants using \p Opcode operation
///
/// \param Opcode the opcode of the binary operator.
/// \param Signed whether the operands are signed or not.
/// \param Op1 the first operand.
/// \param Op2 the second operand.
/// \param T the type of the operands the result.
/// \param DL the DataLayout to compute the result.
/// \return the result of the operation.
static uint64_t combineImpl(unsigned Opcode,
                            bool Signed,
                            Constant *Op1,
                            Constant *Op2,
                            IntegerType *T,
                            const DataLayout &DL) {
  auto *R = ConstantFoldInstOperands(Opcode, T, { Op1, Op2 }, DL);
  return getExtValue(R, Signed, DL);
}

static uint64_t combineImpl(unsigned Opcode,
                            bool Signed,
                            uint64_t Op1,
                            Constant *Op2,
                            IntegerType *T,
                            const DataLayout &DL) {
  return combineImpl(Opcode, Signed, CI::get(T, Op1, Signed), Op2, T, DL);
}

static uint64_t combineImpl(unsigned Opcode,
                            bool Signed,
                            Constant *Op1,
                            uint64_t Op2,
                            IntegerType *T,
                            const DataLayout &DL) {
  return combineImpl(Opcode, Signed, Op1, CI::get(T, Op2, Signed), T, DL);
}

uint64_t BoundedValue::performOp(uint64_t Op1,
                                 unsigned Opcode,
                                 uint64_t Op2,
                                 const DataLayout &DL) const {
  assert(Value != nullptr);

  // Obtain the type
  IntegerType *Ty = dyn_cast<IntegerType>(Value->getType());

  // If it's not an integer type it must be a Store instruction
  if (Ty == nullptr) {
    auto *Store = cast<StoreInst>(Value);
    Ty = cast<IntegerType>(Store->getValueOperand()->getType());
  }

  // Build operands
  bool IsSigned = isSigned();
  auto *COp1 = CI::get(Ty, Op1, IsSigned);
  auto *COp2 = CI::get(Ty, Op2, IsSigned);

  // Compute the result
  auto *Result = ConstantFoldInstOperands(Opcode, Ty, { COp1, COp2 }, DL);
  return getExtValue(Result, IsSigned, DL);
}

BoundedValue BoundedValue::moveTo(llvm::Value *V,
                                  const DataLayout &DL,
                                  uint64_t Offset,
                                  uint64_t Multiplier) const {
  BoundedValue Result = *this;
  Result.Value = V;

  using I = Instruction;
  if (Result.LowerBound != Result.lowerExtreme()) {
    Result.LowerBound = performOp(Result.LowerBound, I::Mul, Multiplier, DL);
    Result.LowerBound = performOp(Result.LowerBound, I::Add, Offset, DL);
  }

  if (Result.UpperBound != Result.upperExtreme()) {
    Result.UpperBound = performOp(Result.UpperBound, I::Mul, Multiplier, DL);
    Result.UpperBound = performOp(Result.UpperBound, I::Add, Offset, DL);
  }

  return Result;
}

bool OSR::combine(unsigned Opcode,
                  Constant *Operand,
                  unsigned FreeOpIndex,
                  const DataLayout &DL) {
  using I = Instruction;
  auto *TheType = cast<IntegerType>(Operand->getType());
  bool Multiplicative = !(Opcode == I::Add || Opcode == I::Sub);
  bool Signed = (Opcode == I::SDiv || Opcode == I::AShr);

  Operand = getConstValue(Operand, DL);

  uint64_t OldValue = Base;
  uint64_t OldFactor = Factor;

  bool Changed = false;

  // Handle the only case of non-commutative operation with first operand
  // constant that we handle: subtraction
  if (!I::isCommutative(Opcode) && FreeOpIndex != 0) {
    assert(Opcode == I::Sub);
    // c - x
    // x = a + b * y
    // (c - a) + (-b) * y
    Base = combineImpl(Opcode, Signed, Operand, Base, TheType, DL);
    Changed |= Base != OldValue;
    auto *MinusOne = Constant::getAllOnesValue(TheType);
    Factor = combineImpl(I::Mul, Signed, MinusOne, Factor, TheType, DL);
    Changed |= OldFactor != Factor;
  } else {
    // Commutative/second operand constant case
    Base = combineImpl(Opcode, Signed, Base, Operand, TheType, DL);
    Changed |= Base != OldValue;

    if (Multiplicative) {
      Factor = combineImpl(Opcode, Signed, Factor, Operand, TheType, DL);
      Changed |= OldFactor != Factor;

    }
  }

  return Changed;
}

class OSRAnnotationWriter : public AssemblyAnnotationWriter {
public:
  OSRAnnotationWriter(OSRAPass &JTFC) : JTFC(JTFC) { }

  virtual void emitInstructionAnnot(const Instruction *I,
                                    formatted_raw_ostream &Output) {
    JTFC.describe(Output, I);
  }

  virtual void emitBasicBlockStartAnnot(const BasicBlock *BB,
                                        formatted_raw_ostream &Output) {
    JTFC.describe(Output, BB);
  }

private:
  OSRAPass &JTFC;
};

void OSR::describe(formatted_raw_ostream &O) const {
  O << "[" << static_cast<int64_t>(Base)
    << " + " << static_cast<int64_t>(Factor) << " * x, with x = ";
  if (BV == nullptr)
    O << "null";
  else
    BV->describe(O);
  O << "]";
}

void BoundedValue::describe(formatted_raw_ostream &O) const {
  if (Negated)
    O << "NOT ";

  O << "(";
  O << getName(Value);
  O << ", ";

  switch (Sign) {
  case AnySignedness:
    O << "*";
    break;
  case UnknownSignedness:
    O << "?";
    break;
  case Signed:
    O << "s";
    break;
  case Unsigned:
    O << "u";
    break;
  case InconsistentSignedness:
    O << "x";
    break;
  }

  if (Bottom) {
    O << ", bottom";
  } else if (!isUninitialized()) {
    O << ", ";
    if (!isConstant() && LowerBound == lowerExtreme()) {
      O << "min";
    } else {
      O << LowerBound;
    }

    O << ", ";

    if (!isConstant() && UpperBound == upperExtreme()) {
      O << "max";
    } else {
      O << UpperBound;
    }
  }

  O << ")";
}

void OSRAPass::describe(formatted_raw_ostream &O,
                        const BasicBlock *BB) const {
  BVs.describe(O, BB);
}

void OSRAPass::describe(formatted_raw_ostream &O,
                        const Instruction *I) const {
  auto OSRIt = OSRs.find(I);
  auto ConstraintsIt = Constraints.find(I);

  if (OSRIt == OSRs.end() && ConstraintsIt == Constraints.end())
    return;

  if (OSRIt != OSRs.end()) {
    O << "  ; ";
    OSRIt->second.describe(O);
    O << "\n";
  }

  if (ConstraintsIt != Constraints.end()) {
    O << "  ;";
    for (auto Constraint : ConstraintsIt->second) {
      O << " ";
      Constraint.describe(O);
    }
    O << "\n";
  }

  if (auto *Load = dyn_cast<LoadInst>(I)) {
    auto LoadReachersIt = LoadReachers.find(Load);
    if (LoadReachersIt != LoadReachers.end()) {
      O << "  ; ";
      for (auto P : LoadReachersIt->second) {
        O << "{" << getName(P.first) << ", ";
        P.second.describe(O);
        O << "} ";
      }
      O << "\n";
    }
  }
}

Constant *OSR::solveEquation(Constant *KnownTerm,
                             bool CeilingRounding,
                             const DataLayout &DL) {
  // (KnownTerm - Base) udiv Factor
  bool IsSigned = BV->isSigned();

  auto *BaseConst = CI::get(KnownTerm->getType(), Base, IsSigned);
  auto *Numerator = CE::getSub(KnownTerm, BaseConst);
  auto *Denominator = CI::get(KnownTerm->getType(), Factor, IsSigned);

  Constant *Remainder = nullptr;
  Constant *Division = nullptr;
  if (IsSigned) {
    Remainder = CE::getSRem(Numerator, Denominator);
    Division = CE::getSDiv(Numerator, Denominator);
  } else {
    Remainder = CE::getURem(Numerator, Denominator);
    Division = CE::getUDiv(Numerator, Denominator);
  }

  if (isa<UndefValue>(Division))
    return Division;

  bool HasRemainder = getConstValue(Remainder, DL)->getLimitedValue() != 0;
  if (CeilingRounding && HasRemainder)
    Division = CE::getAdd(Division, CI::get(Division->getType(), 1));

  return Division;
}

OSR OSRAPass::createOSR(Value *V, BasicBlock *BB) {
  auto OtherOSRIt = OSRs.find(V);
  if (OtherOSRIt != OSRs.end())
    return switchBlock(OtherOSRIt->second, BB);
  else
    return OSR(&BVs.get(BB, V));
}

/// Helper function to check if two BV vectors are identical
static bool differ(SmallVector<BoundedValue, 2> &Old,
                   SmallVector<BoundedValue, 2> &New) {
  if (Old.size() != New.size())
    return true;

  for (auto &OldConstraint : Old) {
    bool Found = false;
    for (auto &NewConstraint : New) {
      if (OldConstraint.value() == NewConstraint.value()) {
        Found = true;
        if (!(OldConstraint == NewConstraint))
          return true;
      }
    }

    if (!Found)
      return true;
  }

  return false;
}

template<BoundedValue::MergeType MT>
static bool mergeBVVectors(OSRAPass::BVVector &Base,
                           OSRAPass::BVVector &New,
                           const DataLayout &DL,
                           Type *Int64) {
  bool Result = false;
  // Merge the two BV vectors
  for (auto &NewConstraint : New) {
    bool Found = false;
    for (auto &BaseConstraint : Base) {
      if (NewConstraint.value() == BaseConstraint.value()) {
        Result |= BaseConstraint.merge<MT>(NewConstraint, DL, Int64);
        Found = true;
        break;
      }
    }

    if (!Found) {
      Result = true;
      Base.push_back(NewConstraint);
    }
  }
  return Result;
}

/// Given an instruction, identifies, if possible, the constant operand.  If
/// both operands are constant, it returns a Constant with the folded operation
/// and nullptr. If only one is constant, it return the constant and a reference
/// to the free operand. If none of the operands are constant returns { nullptr,
/// nullptr }. It also returns { nullptr, nullptr } if I is not commutative and
/// only the first operand is constant.
std::pair<Constant *,
          Value *> OSRAPass::identifyOperands(const Instruction *I,
                                              const DataLayout &DL) {
  assert(I->getNumOperands() == 2);
  Value *FirstOp = I->getOperand(0);
  Value *SecondOp = I->getOperand(1);
  Constant *Constants[2] = {
    dyn_cast<Constant>(FirstOp),
    dyn_cast<Constant>(SecondOp)
  };

  // Is the first operand constant?
  if (auto *Operand = dyn_cast<Instruction>(FirstOp)) {
    auto OSRIt = OSRs.find(Operand);
    if (OSRIt != OSRs.end() && OSRIt->second.isConstant())
      Constants[0] = CI::get(Operand->getType(), OSRIt->second.constant());
  }

  // Is the second operand constant?
  if (auto *Operand = dyn_cast<Instruction>(SecondOp)) {
    auto OSRIt = OSRs.find(Operand);
    if (OSRIt != OSRs.end() && OSRIt->second.isConstant())
      Constants[1] = CI::get(Operand->getType(), OSRIt->second.constant());
  }

  // No constant operands
  if (Constants[0] == nullptr && Constants[1] == nullptr)
    return { nullptr, nullptr };

  // Both operands are constant, constant fold them
  if (Constants[0] != nullptr && Constants[1] != nullptr) {
    Instruction *Clone = I->clone();
    Clone->setOperand(0, Constants[0]);
    Clone->setOperand(1, Constants[1]);
    Constant *Result = ConstantFoldInstruction(Clone, DL);
    if (isa<UndefValue>(Result))
      return { nullptr, nullptr };
    else
      return { Result, nullptr };
  }

  // Only one operand is constant
  if (Constants[0] != nullptr)
    return { Constants[0], SecondOp };
  else
    return { Constants[1], FirstOp };
}

// TODO: check also undefined behaviors due to shifts
static bool isSupportedOperation(unsigned Opcode,
                                 Constant *ConstantOp,
                                 unsigned FreeOpIndex,
                                 const DataLayout &DL) {
  // Division by zero
  if ((Opcode == Instruction::SDiv
       || Opcode == Instruction::UDiv)
      && getZExtValue(ConstantOp, DL) == 0)
    return false;

  // Shift too much
  auto *OperandTy = dyn_cast<IntegerType>(ConstantOp->getType());
  if ((Opcode == Instruction::Shl
       || Opcode == Instruction::LShr
       || Opcode == Instruction::AShr)
      && getZExtValue(ConstantOp, DL) >= OperandTy->getBitWidth())
    return false;

  // 128-bit operand
  auto *ConstantOpTy = dyn_cast<IntegerType>(ConstantOp->getType());
  if (ConstantOpTy != nullptr && ConstantOpTy->getBitWidth() > 64)
    return false;

  if (!Instruction::isCommutative(Opcode)
      && FreeOpIndex != 0
      && Opcode != Instruction::Sub)
    return false;

  return true;
}

bool OSRAPass::updateLoadReacher(LoadInst *Load, Instruction *I, OSR NewOSR) {
  // Check if the instruction propagating the OSR is already a
  // component of this load or not
  auto ReachersIt = LoadReachers.find(Load);
  if (ReachersIt != LoadReachers.end()) {
    auto &Reachers = ReachersIt->second;
    auto Pred = [I] (const std::pair<Instruction *, OSR> &P) {
      return P.first == I;
    };
    auto ReacherIt = std::find_if(Reachers.begin(), Reachers.end(), Pred);
    if (ReacherIt != Reachers.end()) {
      // We've already propagated I to Load in the past, check if we have new
      // information
      if (ReacherIt->second == NewOSR) {
        return false;
      } else {
        *ReacherIt = make_pair(I, NewOSR);
        return true;
      }
    }
  }

  LoadReachers[Load].push_back({ I, NewOSR });

  return true;
}

bool OSRAPass::isDead(Instruction *I) const {
  while (I != nullptr) {
    if (!I->hasOneUse())
      return false;

    auto *U = dyn_cast<Instruction>(*I->user_begin());
    if (U == nullptr)
      return false;

    switch (U->getOpcode()) {
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::IntToPtr:
    case Instruction::PtrToInt:
      I = dyn_cast<Instruction>(U);
      break;
    case Instruction::Store:
      {
        auto *Store = cast<StoreInst>(U);
        if (Store->getValueOperand() != I)
          return false;

        bool Used = false;
        auto *State = dyn_cast<GlobalVariable>(Store->getPointerOperand());
        if (State == nullptr)
          return false;

        auto Visitor = [State, &Used] (BasicBlockRange R) {
          for (Instruction &I : R) {

            if (auto *Load = dyn_cast<LoadInst>(&I)) {
              if (Load->getPointerOperand() == State) {
                Used = true;
                return StopNow;
              }
            } else if (auto *Store = dyn_cast<StoreInst>(&I)) {
              if (Store->getPointerOperand() == State) {
                return NoSuccessors;
              }
            }

          }

          return Continue;
        };
        visitSuccessors(Store, BlockBlackList, Visitor);

        return !Used;
      }
    default:
      return false;
    }
  }

  return false;
}

void OSRAPass::mergeLoadReacher(LoadInst *Load) {
  auto &Reachers = LoadReachers[Load];
  assert(Reachers.size() > 0);

  OSRs.erase(Load);

  // TODO: implement a real merge strategy, considering input boundaries
  OSR Result = Reachers[0].second;
  for (auto P : skip(1, Reachers)) {
    OSR ReachingOSR = P.second;
    if (ReachingOSR != Result) {
      OSR FreeOSR = createOSR(Load, Load->getParent());
      if (Reachers.size() == RDP->getReachingDefinitionsCount(Load))
        BVs.forceBV(Load, pathSensitiveMerge(Load));
      OSRs.insert({ Load, FreeOSR });
      return;
    }
  }

  OSRs.insert({ Load, Result });
  return;
}

/// \brief State of a definition reaching a load while being processed by
///        OSRAPass::pathSensitiveMerge
class Reacher {
public:
  Reacher(LoadInst *Reached,
          Instruction *Reacher,
          OSR &ReachingOSR) :
    Summary(BoundedValue(ReachingOSR.boundedValue()->value())),
    LastMergeHeight(0),
    ReachingOSR(ReachingOSR),
    LTR(std::set<BasicBlock *> { Reacher->getParent() }),
    LastActiveHeight(Active) { }

  /// \brief Notify that the stack has grown
  void newHeight(unsigned NewHeight) {
    LastActiveHeight = std::min(LastActiveHeight, NewHeight);
    LastMergeHeight = std::min(LastMergeHeight, NewHeight);
  }

  /// \brief Check if the reacher is active at the current stack height
  bool isActive(unsigned CurrentHeight) const {
    return CurrentHeight <= LastActiveHeight;
  }

  /// \brief Check if \p BB leads to the definition represented by this object
  bool isLTR(BasicBlock *BB) const {
    return LTR.count(BB) != 0;
  }

  /// \brief Register \p BB as a basic block leading to this definition
  bool registerLTR(BasicBlock *BB) {
    return LTR.insert(BB).second;
  }

  /// \brief Mark this Reacher as active at the current height
  void setActive() {
    LastActiveHeight = Active;
  }

  /// \brief Mark this Reacher as inactive at height \p Height
  void setInactive(unsigned Height) {
    LastActiveHeight = Height;
  }

  /// \brief Set the last height of the stack when a merge was performed
  void setLastMerge(unsigned Height) {
    LastMergeHeight = Height;
  }

  /// \brief Retrieve the last height of the stack when a merge was performed
  unsigned lastMerge() const { return LastMergeHeight; }

  /// Compute a BV relative to \p V by applying the OSR associated to this
  /// definition and the constraints accumulated in Summary
  BoundedValue computeBV(Value *V, const DataLayout &DL, Type *Int64) const {
    auto Result = ReachingOSR.apply(Summary, V, DL);
    if (!Result.isUninitialized() && !Result.isBottom()) {
      using Cmp = CmpInst;
      auto Predicate = Result.isSigned() ? Cmp::ICMP_SLE : Cmp::ICMP_ULE;
      Constant *Compare = CE::getCompare(Predicate,
                                         Result.lower(Int64),
                                         Result.upper(Int64));
      if (getZExtValue(Compare, DL) == 0)
        Result.setBottom();
    }

    return Result;
  }

  /// \brief Rreturn the OSR associated to this definition
  const OSR &osr() const { return ReachingOSR; }

public:
  BoundedValue Summary;   ///< BV representing the known constraints on the
                          ///  reaching definition's value

private:
  unsigned LastMergeHeight;
  OSR &ReachingOSR;
  const unsigned Active = std::numeric_limits<unsigned>::max();
  std::set<BasicBlock *> LTR;
  unsigned LastActiveHeight;
};

BoundedValue OSRAPass::pathSensitiveMerge(LoadInst *Reached) {
  // Initialization steps
  const unsigned MaxDepth = 5;
  Module *M = Reached->getParent()->getParent()->getParent();
  const DataLayout &DL = M->getDataLayout();
  Type *Int64 = IntegerType::get(M->getContext(), 64);
  MemoryAccess ReachedMA(Reached, DL);

  // Debug support
  raw_os_ostream OsOstream(dbg);
  formatted_raw_ostream FormattedStream(OsOstream);
  FormattedStream.SetUnbuffered();

  DBG("psm", dbg << "Performing PSM for " << getName(Reached) << "\n";);

  std::vector<Reacher> Reachers;
  Reachers.reserve(LoadReachers[Reached].size());
  unsigned ReacherIndex = 0;
  for (auto &P : LoadReachers[Reached]) {
    ReacherIndex++;
    // TODO: isConstant?
    if (P.second.factor() == 0)
      return BoundedValue(Reached);
    Reachers.emplace_back(Reached, P.first, P.second);
    DBG("psm", dbg << "  Reacher " << std::dec << ReacherIndex
        << " is " << getName(P.first) << "\n";);
  }
  assert(Reachers.size() > 0);

  struct State {
    BasicBlock *BB;
    pred_iterator PredecessorIt;
  };
  std::set<BasicBlock *> InStack;
  std::vector<State> Stack;
  State Initial = {
    Reached->getParent(),
    getValidPred(Reached->getParent()),
  };
  if (Initial.PredecessorIt == pred_end(Initial.BB))
    return BoundedValue(Reached);

  Stack.push_back(Initial);
  InStack.insert(Reached->getParent());

  while (!Stack.empty()) {
    State &S = Stack.back();
    unsigned Height = Stack.size();
    BasicBlock *Pred = *S.PredecessorIt;

    DBG("psm", dbg << "Exploring " << getName(Pred) << "\n";);

    // Check if any store in Pred can alias ReachedMA
    bool MayAlias = MemoryAccess::mayAlias(Pred, ReachedMA, DL);

    // Hold whether we should proceed to the predecessors or not
    // Initialize to false, the code handling the various reacher will enable
    // this flag if at least one of the reachers is active
    bool Proceed = false;

    // Reacher-specific handling
    ReacherIndex = 0;
    for (Reacher &R : Reachers) {
      ReacherIndex++;

      // Check if this reacher has been deactivated
      if (!R.isActive(Height))
        continue;

      // Is this a BB leading to the reacher?
      if (R.isLTR(Pred)) {
        DBG("psm", dbg << "  Merging reacher " << ReacherIndex << "\n";);

        // Insert everything is on the stack, but stop if we meet one that's
        // already there
        for (State &NewLTRState : Stack)
          if (!R.registerLTR(NewLTRState.BB))
            break;

        // Perform merge from the top to last merge height
        BoundedValue Result = R.Summary;
        auto Range = make_range(Stack.begin() + R.lastMerge(), Stack.end());
        for (State &ToMerge : Range) {
          // Obtain the constraint from the appropriate edge
          BoundedValue *EdgeBV = BVs.getEdge(ToMerge.BB,
                                             *ToMerge.PredecessorIt,
                                             Result.value());

          // if (EdgeBV == nullptr)
          //   return BoundedValue(Reached);

          if (EdgeBV != nullptr) {
            // And-merge
            Result.merge<BoundedValue::And>(*EdgeBV, DL, Int64);

            DBG("psm", {
                dbg << "    Got ";
                EdgeBV->describe(FormattedStream);
                dbg << " from the " << getName(*ToMerge.PredecessorIt)
                    << " -> " << getName(ToMerge.BB)
                    << " edge: ";
                Result.describe(FormattedStream);
                dbg << "\n";
              });

            if (Result.isBottom())
              break;

          } else {
            DBG("psm", {
                dbg << "    Got no info";
                dbg << " from the " << getName(*ToMerge.PredecessorIt)
                    << " -> " << getName(ToMerge.BB)
                    << " edge\n";
              });
          }
        }

        // If result is bottom, we went through a contradictory branch, ignore
        // it and deactivate
        if (!Result.isBottom()) {
          R.Summary = Result;

          // Register the current height as the last merge
          R.setLastMerge(Height);
        } else {
          DBG("psm", dbg << "We got an incoherent situation, ignore it\n";);
        }

        // Deactivate
        R.setInactive(Height);
      } else if (MayAlias) {
        DBG("psm", dbg << "  Deactivating reacher " << ReacherIndex << "\n";);

        // We don't know if it's an LTR, check if it may alias, and if so,
        // deactivate this reacher
        R.setInactive(Height);
      } else {
        // Activate
        R.setActive();

        // At least one of the reacher is active, we have to proceed to the
        // predecessor
        Proceed = true;
      }
    }

    // Check it's not already in stack
    Proceed &= InStack.count(Pred) == 0;

    // Check we're not exceeding the maximum allowed depth
    Proceed &= Height < MaxDepth;

    // Check we have at least a non-dispatcher predecessor
    pred_iterator NewPredIt = getValidPred(Pred);
    Proceed &= NewPredIt != pred_end(Pred);

    if (Proceed) {
      // We have to go deeper
      State NewState = {
        Pred,
        NewPredIt
      };
      Stack.push_back(NewState);
      InStack.insert(Pred);
    } else {
      // Pop until the stack is empty or we still have unexplored predecessors
      unsigned OldHeight = Stack.size();
      while (Stack.size() != 0) {
        State &Top = Stack.back();
        auto End = pred_end(Top.BB);
        if (nextValidPred(++Top.PredecessorIt, End) != End)
          break;

        InStack.erase(Top.BB);
        Stack.pop_back();
      }

      // If we popped something make sure we update all the heights
      unsigned NewHeight = Stack.size();
      if (NewHeight < OldHeight)
        for (Reacher &R : Reachers)
          R.newHeight(NewHeight);
    }

  }

  // Or-merge all the collected BVs
  // TODO: adding the OSR offset is safe, but the multiplier?
  BoundedValue FinalBV = Reachers[0].computeBV(Reached, DL, Int64);

  for (Reacher &R : skip(1, Reachers)) {
    BoundedValue ReacherBV = R.computeBV(Reached, DL, Int64);

    DBG("psm", {
        dbg << "";
        FinalBV.describe(FormattedStream);
        dbg << " += ";
        ReacherBV.describe(FormattedStream);
        dbg << " (from ";
        R.osr().describe(FormattedStream);
        dbg << ")\n";
      });

    if (FinalBV.isBottom())
      return BoundedValue(Reached);

    FinalBV.merge<BoundedValue::Or>(ReacherBV, DL, Int64);
  }

  if (FinalBV.isUninitialized() || FinalBV.isTop() || FinalBV.isBottom())
    return BoundedValue(Reached);

  DBG("psm", {
      dbg << "FinalBV: ";
      FinalBV.describe(FormattedStream);
      dbg << "\n";
    });

  assert(!FinalBV.isUninitialized());
  return FinalBV;
}

// Terminology:
// * OSR: Offseted Shifted Range, our main data flow value which represents the
//        result of an instruction as another value, which lies withing a
//        certain range of values, multiplied by a factor and with an
//        offset, e.g. 100 + 4 * x, with 0 < x < 4.
// * free value: a value we can't represent as an OSR of another value
// * bounded variable (or BV): a free value and the range within which it lies.
bool OSRAPass::runOnFunction(Function &F) {
  DBG("passes", { dbg << "Starting OSRAPass\n"; });

  const DataLayout DL = F.getParent()->getDataLayout();
  RDP = &getAnalysis<ConditionalReachedLoadsPass>();
  auto &SCP = getAnalysis<SimplifyComparisonsPass>();

  // The Overtaken map keeps track of which load/store instructions have been
  // overtaken by another load/store, meaning that they are not "free" but can
  // be expressed in terms of another stored/loaded value
  std::map<const Value *, const Value *> Overtaken;

  auto *Int64 = Type::getInt64Ty(F.getParent()->getContext());
  using UpdateFunc = std::function<BVVector(BVVector &)>;

  for (auto &BB : F) {
    if (!BB.empty()) {
      if (auto *Call = dyn_cast<CallInst>(&*BB.begin())) {
        Function *Callee = Call->getCalledFunction();
        // TODO: comparing with "newpc" string is sad
        if (Callee != nullptr && Callee->getName() == "newpc")
          break;
      }
    }
    BlockBlackList.insert(&BB);
  }

  // Cleanup all the data
  freeContainer(OSRs);
  BVs = BVMap(&BlockBlackList, &DL, Int64);
  freeContainer(Constraints);

  // Initialize the WorkList with all the instructions in the function
  UniquedQueue<Instruction *> WorkList;
  auto &BBList = F.getBasicBlockList();
  for (auto &BB : make_range(BBList.begin(), BBList.end()))
    if (BlockBlackList.find(&BB) == BlockBlackList.end())
      for (auto &I : make_range(BB.begin(), BB.end()))
        WorkList.insert(&I);

  // TODO: make these member functions
  auto InBlackList = [this] (BasicBlock *BB) {
    return BlockBlackList.find(BB) != BlockBlackList.end();
  };

  auto EnqueueUsers = [this, &WorkList] (Instruction *I) {
    for (User *U : I->users())
      if (auto *UI = dyn_cast<Instruction>(U))
        if (BlockBlackList.find(UI->getParent()) == BlockBlackList.end()) {
          WorkList.insert(UI);
        }
  };

  auto PropagateConstraints = [this, &EnqueueUsers] (Instruction *I,
                                                     Value *Operand,
                                                     UpdateFunc Updater) {
    // We want to propagate contraints through zero-extensions
    if (auto *OperandInst = dyn_cast<Instruction>(Operand)) {
      auto OperandConstraintIt = Constraints.find(OperandInst);
      auto InstrConstraintIt = Constraints.find(I);

      // Does the operand have constraints?
      if (OperandConstraintIt != Constraints.end()) {
        auto New = Updater(OperandConstraintIt->second);

        // Does the instruction already had a constraint?
        if (InstrConstraintIt != Constraints.end()) {
          // Did the constraint changed?
          if (!differ(New, InstrConstraintIt->second))
            return;

          Constraints.erase(InstrConstraintIt);
        }

        Constraints.insert({ I, New });
        EnqueueUsers(I);
      }
    }
  };

  while (!WorkList.empty()) {
    Instruction *I = WorkList.pop();

    // TODO: create a member function for each group of opcodes
    unsigned Opcode = I->getOpcode();
    switch (Opcode) {
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Mul:
    case Instruction::Shl:
    case Instruction::SDiv:
    case Instruction::UDiv:
    case Instruction::LShr:
    case Instruction::AShr:
      {
        // Check if it's a free value
        auto OldOSRIt = OSRs.find(I);
        bool IsFree = OldOSRIt == OSRs.end();
        bool Changed = false;

        Constant *ConstantOp = nullptr;
        Value *OtherOp = nullptr;
        std::tie(ConstantOp, OtherOp) = identifyOperands(I, DL);

        if (OtherOp == nullptr) {
          if (ConstantOp != nullptr) {
            // If OtherOp is nullptr but ConstantOp is not it means we were able
            // to fold the operation in a constant
            if (!IsFree)
              OSRs.erase(I);

            uint64_t Constant = getZExtValue(ConstantOp, DL);
            BoundedValue ConstantBV = BoundedValue::createConstant(I, Constant);
            auto &BV = BVs.forceBV(I, ConstantBV);
            OSR ConstantOSR(&BV);
            OSRs.emplace(make_pair(I, ConstantOSR));
            EnqueueUsers(I);
          }

          // In any case, break
          break;
        }

        // Get or create an OSR for the non-constant operator, this
        // will be our starting point
        OSR NewOSR = createOSR(OtherOp, I->getParent());
        if (!IsFree) {
          if (NewOSR.isRelativeTo(OldOSRIt->second.boundedValue()->value())) {
            break;
          } else {
            Changed = true;
          }
        }

        // Check we're not depending on ourselves, if we are leave us as a free
        // value
        if (NewOSR.isRelativeTo(I)) {
          assert(IsFree);
          break;
        }

        // TODO: this is probably a bad idea
        if (NewOSR.boundedValue()->isBottom()) {
          if (!IsFree)
            OSRs.erase(OldOSRIt);
          break;
        }

        // TODO: skip this if isDead(I)
        // Update signedness information if the given operation is
        // sign-aware
        if (Opcode == Instruction::SDiv
            || Opcode == Instruction::UDiv
            || Opcode == Instruction::LShr
            || Opcode == Instruction::AShr) {
          BVs.setSignedness(I->getParent(),
                            NewOSR.boundedValue()->value(),
                            Opcode == Instruction::SDiv
                            || Opcode == Instruction::AShr);
        }

        // Check for undefined behaviors
        unsigned FreeOpIndex = OtherOp == I->getOperand(0) ? 0 : 1;
        if (!isSupportedOperation(Opcode, ConstantOp, FreeOpIndex, DL)) {
          NewOSR = OSR(&BVs.get(I->getParent(), I));
          Changed = true;
        } else {
          // Combine the base OSR with the new operation
          Changed |= NewOSR.combine(Opcode, ConstantOp, FreeOpIndex, DL);
        }

        // Check if the OSR has changed
        if (IsFree || Changed) {
          // Update the OSR and enqueue all I's uses
          if (!IsFree)
            OSRs.erase(I);
          OSRs.emplace(make_pair(I, NewOSR));
          EnqueueUsers(I);
        }

        break;
      }
    case Instruction::ICmp:
      {
        // TODO: this part is quite ugly, try to improve it
        auto SimplifiedComparison = SCP.getComparison(cast<CmpInst>(I));
        ICmpInst *Comparison = new ICmpInst(SimplifiedComparison.Predicate,
                                            SimplifiedComparison.LHS,
                                            SimplifiedComparison.RHS);
        std::unique_ptr<ICmpInst> SimplifiedCmpInst(Comparison);

        Predicate P = Comparison->getPredicate();

        Value *LHS = Comparison->getOperand(0);
        Value *RHS = Comparison->getOperand(1);

        Constant *ConstOp = nullptr;
        Value *FreeOpValue = nullptr;
        Instruction *FreeOp = nullptr;
        std::tie(ConstOp, FreeOpValue) = identifyOperands(Comparison, DL);
        if (FreeOpValue != nullptr) {
          FreeOp = dyn_cast<Instruction>(FreeOpValue);
          if (FreeOp == nullptr)
            break;
        }

        if (isDead(I))
          break;

        // Comparison for equality and inequality are handled to propagate
        // constraints in case of test of the result of a comparison (e.g., (x <
        // 3) == 0).
        if (ConstOp != nullptr && FreeOp != nullptr
            && Constraints.find(FreeOp) != Constraints.end()
            && (P == CmpInst::ICMP_EQ || P == CmpInst::ICMP_NE)) {
          // If we're comparing with 0 for equality or inequality and the
          // non-constant operand has constraints, propagate them flipping them
          // (if necessary).
          if (getZExtValue(ConstOp, DL) == 0) {

            if (P == CmpInst::ICMP_EQ) {
              PropagateConstraints(I, FreeOp, [] (BVVector &Constraints) {
                  BVVector Result = Constraints;
                  // TODO: This is wrong! !(a & b) == !a || !b,
                  //       not !a && !b
                  for (auto &Constraint : Result)
                    Constraint.flip();
                  return Result;
                });
            } else {
              PropagateConstraints(I, FreeOp, [] (BVVector &Constraints) {
                  return Constraints;
                });
            }

            // Do not proceed
            break;
          }
        }

        // Compute a new constraint
        // Check the comparison operator is a supported one
        if (P != CmpInst::ICMP_UGT
            && P != CmpInst::ICMP_UGE
            && P != CmpInst::ICMP_SGT
            && P != CmpInst::ICMP_SGE
            && P != CmpInst::ICMP_ULT
            && P != CmpInst::ICMP_ULE
            && P != CmpInst::ICMP_SLT
            && P != CmpInst::ICMP_SLE
            && P != CmpInst::ICMP_EQ
            && P != CmpInst::ICMP_NE)
          break;

        auto OldBVsIt = Constraints.find(I);
        bool HasConstraints = OldBVsIt != Constraints.end();
        BVVector NewConstraints;

        if (FreeOp == nullptr) {
          if (ConstOp == nullptr) {
            // Both operands are free, give up

            // TODO: are we sure this is what we want?
            if (HasConstraints)
              Constraints.erase(OldBVsIt);
            HasConstraints = false;
            break;
          } else {
            // FreeOpValue is nullptr but ConstOp is not: we were able to fold
            // the operation into a constant

            if (getZExtValue(ConstOp, DL) != 0) {
              // The comparison holds, we're saying nothing useful (e.g. 2 < 3),
              // remove any constraint
              if (HasConstraints)
                Constraints.erase(OldBVsIt);
              HasConstraints = false;
            } else {
              // The comparison does not hold, move to bottom all the involved
              // BVs

              auto *FirstOp = dyn_cast<Instruction>(LHS);
              if (FirstOp != nullptr) {
                auto FirstOSRIt = OSRs.find(FirstOp);
                if (FirstOSRIt != OSRs.end()) {
                  auto FirstOSR = FirstOSRIt->second;
                  NewConstraints.push_back(*FirstOSR.boundedValue());
                }
              }

              if (auto *SecondOp = dyn_cast<Instruction>(RHS)) {
                auto SecondOSRIt = OSRs.find(SecondOp);
                if (SecondOSRIt != OSRs.end()) {
                  auto SecondOSR = SecondOSRIt->second;
                  NewConstraints.push_back(*SecondOSR.boundedValue());
                }
              }

              for (auto &Constraint : NewConstraints)
                Constraint.setBottom();

            }
          }

        } else {
          // We have a constant operand and a free one

          BasicBlock *BB = I->getParent();

          auto Handle = [&] (OSR &BaseOp) {
            if (BaseOp.boundedValue()->isBottom()
                || BaseOp.isRelativeTo(I)
                || BaseOp.factor() == 0)
              return;


            // Notify the BV about the sign we're going to use, unless it's a
            // comparison of (in)equality
            bool IsSigned;
            if (P != CmpInst::ICMP_EQ && P != CmpInst::ICMP_NE) {
              IsSigned = Comparison->isSigned();
              BVs.setSignedness(BB, BaseOp.boundedValue()->value(), IsSigned);
            } else {
              // TODO: we don't know what sign to use here, so we ignore it,
              //       should we switch to AnySignedness?
              if (BaseOp.boundedValue()->isUninitialized())
                return;

              IsSigned = BaseOp.boundedValue()->isSigned();
            }

            // Setting the sign might lead to bottom
            if (BaseOp.boundedValue()->isBottom())
              return;

            // Create a copy of the current value of the BV
            BoundedValue NewBV = *(BaseOp.boundedValue());

            // Solve the equation to obtain the new boundary value
            // x <  1.5 == x <  2 (Ceiling)
            // x <= 1.5 == x <= 1 (Floor)
            // x >  1.5 == x >  1 (Floor)
            // x >= 1.5 == x >= 2 (Ceiling)
            bool RoundUp = (P == CmpInst::ICMP_UGE
                            || P == CmpInst::ICMP_SGE
                            || P == CmpInst::ICMP_ULT
                            || P == CmpInst::ICMP_SLT);

            Constant *NewBoundC = BaseOp.solveEquation(ConstOp, RoundUp, DL);
            if (isa<UndefValue>(NewBoundC))
              return;

            uint64_t NewBound = getExtValue(NewBoundC, IsSigned, DL);

            using BV = BoundedValue;
            switch (P) {
            case CmpInst::ICMP_UGT:
            case CmpInst::ICMP_UGE:
            case CmpInst::ICMP_SGT:
            case CmpInst::ICMP_SGE:
              if (Comparison->isFalseWhenEqual())
                NewBound++;

              NewBV.merge(BV::createGE(NewBV.value(), NewBound, IsSigned),
                          DL, Int64);
              break;
            case CmpInst::ICMP_ULT:
            case CmpInst::ICMP_ULE:
            case CmpInst::ICMP_SLT:
            case CmpInst::ICMP_SLE:
              if (Comparison->isFalseWhenEqual())
                NewBound--;

              NewBV.merge(BV::createLE(NewBV.value(), NewBound, IsSigned),
                          DL, Int64);
              break;
            case CmpInst::ICMP_EQ:
              NewBV.merge(BV::createEQ(NewBV.value(),
                                       NewBound,
                                       NewBV.isSigned()),
                          DL,
                          Int64);
              break;
            case CmpInst::ICMP_NE:
              NewBV.merge(BV::createNE(NewBV.value(),
                                       NewBound,
                                       NewBV.isSigned()),
                          DL,
                          Int64);
              break;
            default:
              assert(false);
              break;
            }

            NewConstraints.push_back(NewBV);
          };

          OSR TheOSR = createOSR(FreeOp, BB);
          NewConstraints.clear();

          // Handle the base case
          Handle(TheOSR);

          // Handle all the reaching definitions, if it's referred to a load
          const Value *BaseValue = nullptr;
          if (TheOSR.boundedValue() != nullptr)
            BaseValue = TheOSR.boundedValue()->value();

          if (BaseValue != nullptr) {
            if (auto *Load = dyn_cast<LoadInst>(BaseValue)) {
              const OSR *LoadOSR = getOSR(Load);
              auto &Reachers = LoadReachers[Load];

              // Register this instruction to be visited again when Load changes
              Subscriptions[Load].insert(I);
              if (Reachers.size() > 1
                  && LoadOSR != nullptr
                  && LoadOSR->boundedValue()->value() == Load) {

                for (auto &P : Reachers) {
                  if (!P.second.isConstant()
                      && P.second.boundedValue() != nullptr
                      && P.second.boundedValue()->value() != nullptr
                      && P.second.boundedValue()->value() != Load) {
                    OSR TheOSR = switchBlock(P.second, BB);
                    Handle(TheOSR);
                  }
                }

              }
            }
          }


        }

        bool Changed = true;

        // Check against the old constraints associated with this comparison
        if (HasConstraints) {
          BVVector &OldBVsVector = OldBVsIt->second;
          if (NewConstraints.size() == OldBVsVector.size()) {
            bool Different = false;
            auto OldIt = OldBVsVector.begin();
            auto NewIt = NewConstraints.begin();

            // Loop over all the elements until a different one is found or we
            // reached the end
            while (!Different && OldIt != OldBVsVector.end()) {
              Different |= *OldIt != *NewIt;
              OldIt++;
              NewIt++;
            }

            Changed = Different;
          }
        }

        // If something changed replace the BV vector and re-enqueue all the
        // users
        if (Changed) {
          Constraints[I] = NewConstraints;
          EnqueueUsers(I);
        }

        break;
      }
    case Instruction::ZExt:
      {
        PropagateConstraints(I, I->getOperand(0), [] (BVVector &BV) {
            return BV;
          });
        break;
      }
    case Instruction::And:
    case Instruction::Or:
      {
        Instruction *FirstOperand = dyn_cast<Instruction>(I->getOperand(0));
        Instruction *SecondOperand = dyn_cast<Instruction>(I->getOperand(1));
        if (FirstOperand == nullptr || SecondOperand == nullptr)
          break;

        auto FirstConstraintIt = Constraints.find(FirstOperand);
        auto SecondConstraintIt = Constraints.find(SecondOperand);

        // We can merge the BVs only if both operands have one
        if (FirstConstraintIt == Constraints.end()
            || SecondConstraintIt == Constraints.end())
          break;

        // Initialize the new boundaries with the first operand
        auto NewConstraints = FirstConstraintIt->second;
        auto &OtherConstraints = SecondConstraintIt->second;

        if (Opcode == Instruction::And)
          mergeBVVectors<AndMerge>(NewConstraints, OtherConstraints, DL, Int64);
        else
          mergeBVVectors<OrMerge>(NewConstraints, OtherConstraints, DL, Int64);

        bool Changed = true;
        // If this instruction already had constraints, compare them with the
        // new ones
        auto OldConstraintsIt = Constraints.find(I);
        if (OldConstraintsIt != Constraints.end())
          Changed = differ(OldConstraintsIt->second, NewConstraints);

        // If something changed, register the new constraints and re-enqueue all
        // the users of the instruction
        if (Changed) {
          Constraints[I] = NewConstraints;
          EnqueueUsers(I);
        }

        break;
      }
    case Instruction::Br:
      {
        auto *Branch = cast<BranchInst>(I);

        // Unconditional branches bring no useful information
        if (Branch->isUnconditional())
          break;

        auto *Condition = dyn_cast<Instruction>(Branch->getCondition());
        if (Condition == nullptr)
          break;

        // Were we able to handle the condition?
        auto BranchConstraintsIt = Constraints.find(Condition);
        if (BranchConstraintsIt == Constraints.end())
          break;

        // Take a reference to the constraints, and produce a complementary
        // version
        auto &BranchConstraints = BranchConstraintsIt->second;
        BVVector FlippedBranchConstraints = BranchConstraintsIt->second;
        // TODO: This is wrong! !(a & b) == !a || !b, not !a && !b
        for (auto &BranchConstraint : FlippedBranchConstraints)
          BranchConstraint.flip();

        // Create and initialize the worklist with the positive constraints for
        // the true branch, and the negated constraints for the false branch
        struct WLEntry {
          WLEntry(BasicBlock *Target,
                  BasicBlock *Origin,
                  BVVector Constraints) :
            Target(Target), Origin(Origin), Constraints(Constraints) { }

          BasicBlock *Target;
          BasicBlock *Origin;
          BVVector Constraints;
        };

        std::vector<WLEntry> ConstraintsWL;
        if (!InBlackList(Branch->getSuccessor(0))) {
          ConstraintsWL.push_back(WLEntry(Branch->getSuccessor(0),
                                          Branch->getParent(),
                                          BranchConstraints));
        }

        if (!InBlackList(Branch->getSuccessor(1))) {
          ConstraintsWL.push_back(WLEntry(Branch->getSuccessor(1),
                                          Branch->getParent(),
                                          FlippedBranchConstraints));
        }

        // TODO: can we do this in a DFA way?
        // Process the worklist
        while (!ConstraintsWL.empty()) {
          auto Entry = ConstraintsWL.back();
          ConstraintsWL.pop_back();
          assert(BlockBlackList.find(Entry.Target) == BlockBlackList.end());

          // Merge each changed bound with the existing one
          for (auto ConstraintIt = Entry.Constraints.begin();
               ConstraintIt != Entry.Constraints.end();) {
            auto Result = BVs.update(Entry.Target, Entry.Origin, *ConstraintIt);
            bool Changed = Result.first;
            BoundedValue &NewBV = Result.second;

            if (Changed) {
              // From now we propagate the updated constraint
              *ConstraintIt = NewBV;
              ConstraintIt++;
            } else {
              ConstraintIt = Entry.Constraints.erase(ConstraintIt);
            }
          }

          // Compute the set of affected values
          llvm::SmallSet<const Value *, 5> Affected;
          for (BoundedValue &Constraint : Entry.Constraints)
            Affected.insert(Constraint.value());

          // Look for instructions using constraints that have changed
          for (Instruction &ConstraintUser : *Entry.Target) {
            // Avoid looking up instructions that simply cannot be there
            switch (ConstraintUser.getOpcode()) {
            case Instruction::ICmp:
            case Instruction::And:
            case Instruction::Or:
              {
                // Ignore instructions without an associated constraint
                auto ConstraintIt = Constraints.find(&ConstraintUser);
                if (ConstraintIt == Constraints.end())
                  continue;

                // If it's using one of the changed variables, insert it in the
                // worklist
                BVVector &InstructionConstraints = ConstraintIt->second;

                for (BoundedValue &Constraint : InstructionConstraints) {
                  if (Affected.count(Constraint.value()) != 0) {
                    WorkList.insert(&ConstraintUser);
                    break;
                  }
                }

                break;
              }
            case Instruction::Load:
              {
                // Check if any of the reaching definitions of this load is
                // affected by the constraints being propagated
                LoadInst *Load = cast<LoadInst>(&ConstraintUser);
                auto ReachersIt = LoadReachers.find(Load);
                if (ReachersIt == LoadReachers.end())
                  break;

                auto &Reachers = ReachersIt->second;

                for (auto &P : Reachers) {
                  const Value *ReacherValue = nullptr;
                  if (P.second.boundedValue() != nullptr)
                    ReacherValue = P.second.boundedValue()->value();

                  if (Affected.count(ReacherValue) != 0) {
                    // We're affected, update
                    mergeLoadReacher(Load);
                    WorkList.insert(Load);
                    EnqueueUsers(Load);
                    Affected.insert(Load);
                    break;
                  }
                }

               break;
              }
            default:
              break;
            }

          }

          // Propagate the new constraints to the successors (except for the
          // dispatcher)
          if (Entry.Constraints.size() != 0)
            for (BasicBlock *Successor : successors(Entry.Target))
              if (BlockBlackList.find(Successor) == BlockBlackList.end())
                ConstraintsWL.push_back(WLEntry(Successor,
                                                Entry.Target,
                                                Entry.Constraints));
        }

        break;
      }
    case Instruction::Store:
    case Instruction::Load:
      {
        // Create the OSR to propagate
        MemoryAccess MA;
        // TODO: rename SelfOSR (it's not always self)
        OSR SelfOSR;
        BVVector TheConstraints;
        bool HasConstraints = false;

        if (auto *TheLoad = dyn_cast<LoadInst>(I)) {
          // It's a load
          MA = MemoryAccess(TheLoad, DL);
          auto OSRIt = OSRs.find(I);
          if (OSRIt != OSRs.end())
            SelfOSR = OSRIt->second;
          else
            SelfOSR = OSR(&BVs.get(I->getParent(), I));

        } else if (auto *TheStore = dyn_cast<StoreInst>(I)) {
          // It's a store
          MA = MemoryAccess(TheStore, DL);
          Value *ValueOp = TheStore->getValueOperand();

          if (auto *ConstantOp = dyn_cast<Constant>(ValueOp)) {

            // We're storing a constant, create a constant OSR
            uint64_t Constant = getZExtValue(ConstantOp, DL);
            BoundedValue ConstantBV = BoundedValue::createConstant(ConstantOp,
                                                                   Constant);
            auto &BV = BVs.forceBV(I->getParent(), ConstantOp, ConstantBV);
            SelfOSR = OSR(&BV);

          } else if (auto *ToStore = dyn_cast<Instruction>(ValueOp)) {

            // Compute the OSR to propagate: either the one of the value to
            // store, or a self-referencing one
            auto OSRIt = OSRs.find(ToStore);
            if (OSRIt != OSRs.end())
              SelfOSR = OSRIt->second;
            else
              SelfOSR = OSR(&BVs.get(I->getParent(), I));

            // Check if the value we're storing has constraints
            auto ConstraintIt = Constraints.find(ToStore);
            HasConstraints = ConstraintIt != Constraints.end();
            if (HasConstraints)
              TheConstraints = ConstraintIt->second;

          }

        }

        auto ReachedLoads = RDP->getReachedLoads(I);
        for (LoadInst *ReachedLoad : ReachedLoads) {

          // OSR propagation first

          // Take the reference OSR (SelfOSR) and "contextualize" it in
          // the reached load's basic block
          OSR NewOSR = switchBlock(SelfOSR, ReachedLoad->getParent());

          bool Changed = updateLoadReacher(ReachedLoad, I, NewOSR);
          if (Changed)
            mergeLoadReacher(ReachedLoad);

          // Constraints propagation
          if (HasConstraints) {
            // Does the reached load carries any constraints already?
            auto ReachedLoadConstraintIt = Constraints.find(ReachedLoad);
            if (ReachedLoadConstraintIt != Constraints.end()) {
              // Merge the constraints (using the `or` logic) directly in-place
              // in the reached load's BVVector
              using BV = BoundedValue;
              Changed |= mergeBVVectors<BV::Or>(ReachedLoadConstraintIt->second,
                                                TheConstraints,
                                                DL,
                                                Int64);
            } else {
              // The reached load has no constraints, simply propagate the input
              // ones
              Constraints.insert({ ReachedLoad, TheConstraints });
              Changed = true;
            }
          }

          // If OSR or constraints have changed, mark the reached load and its
          // uses to be visited again
          if (Changed) {
            WorkList.insert(ReachedLoad);
            EnqueueUsers(ReachedLoad);
            for (Instruction *Subscriber : Subscriptions[ReachedLoad])
              WorkList.insert(Subscriber);
          }

        }

        break;
      }
    default:
      break;
    }
  }

  // Free up memory not part of the analysis result
  freeContainer(Constraints);
  freeContainer(LoadReachers);
  freeContainer(BlockBlackList);
  freeContainer(Subscriptions);

  DBG("osr", {
      BVs.prepareDescribe();
      raw_os_ostream OutputStream(dbg);
      F.getParent()->print(OutputStream, new OSRAnnotationWriter(*this));
    });

  DBG("passes", { dbg << "Ending OSRAPass\n"; });

  return false;
}

void OSRAPass::BVMap::describe(formatted_raw_ostream &O,
                               const BasicBlock *BB) const {
  if (BBMap.find(BB) != BBMap.end())
    for (MapValue &MV : BBMap[BB]) {
      O << "  ; ";

      {
        auto &BVO = MV.Summary;
        O << "<";
        BVO.describe(O);
        O << ">";
      }

      if (MV.Components.size() > 0)
        O << " = ";

      for (auto &BVO : MV.Components) {
        O << "<";
        O << getName(BVO.first);
        O << ", ";
        BVO.second.describe(O);
        O << "> || ";
      }

      O << "\n";
    }
  O << "\n";
}

std::pair<bool, BoundedValue&> OSRAPass::BVMap::update(BasicBlock *Target,
                                                       BasicBlock *Origin,
                                                       BoundedValue NewBV) {
  auto Index = make_pair(Target, NewBV.value());
  auto MapIt = TheMap.find(Index);
  MapValue *BVOVector = nullptr;

  // Have we ever seen this value for this basic block?
  if (MapIt == TheMap.end()) {
    // No, just insert it
    MapValue NewBVOVector;
    NewBVOVector.Components.push_back({ make_pair(Origin, NewBV) });
    BVOVector = &TheMap.insert({ Index, NewBVOVector }).first->second;
    return { true, summarize(Target, BVOVector) };
  } else if (isForced(MapIt)) {
    return { false, MapIt->second.Summary };
  } else {
    bool Changed = true;
    BVOVector = &MapIt->second;

    // Look for an entry with the given origin
    BoundedValue *Base = nullptr;
    for (BVWithOrigin &BVO : BVOVector->Components)
      if (BVO.first == Origin)
        Base = &BVO.second;

    // Did we ever see this Origin?
    if (Base == nullptr)
      BVOVector->Components.push_back({ Origin, NewBV });
    else
      Changed = Base->merge<AndMerge>(NewBV, *DL, Int64);

    // Re-merge all the entries
    auto &Result = summarize(Target, BVOVector);
    return { Changed, Result };
  }

  // TODO: should Changed be false if isForced?

}

BoundedValue &OSRAPass::BVMap::summarize(BasicBlock *Target,
                                         MapValue *BVOVector) {

  if (BVOVector->Components.size() == 0)
    return BVOVector->Summary;

  // Initialize the summary BV with the first BV
  BVOVector->Summary = BVOVector->Components[0].second;

  unsigned PredecessorsCount = 0;
  for (auto *Predecessor : predecessors(Target))
    if (BlockBlackList->find(Predecessor) == BlockBlackList->end()
        && !pred_empty(Predecessor))
      PredecessorsCount++;

  // Do we have a constraint for each predecessor?
  if (BVOVector->Components.size() == PredecessorsCount) {
    // Yes, we can populate the summary by merging all the components
    for (auto &BVO : skip(1, BVOVector->Components))
      BVOVector->Summary.merge<OrMerge>(BVO.second, *DL, Int64);
  } else {
    // No, keep the summary at top
    BVOVector->Summary.setTop();
  }

  return BVOVector->Summary;
}

bool OSR::compare(unsigned short P,
                  Constant *C,
                  const DataLayout &DL,
                  Type *Int64) {
  Constant *BaseConstant = CI::get(Int64, Base);
  Constant *Compare = CE::getCompare(P, BaseConstant, C);
  return getConstValue(Compare, DL)->getLimitedValue() != 0;
}

void BoundedValue::setSignedness(bool IsSigned) {
  // TODO: assert?
  if (Bottom)
    return;

  // If we're already inconsistent just return
  if (Sign == InconsistentSignedness)
    return;

  Signedness NewSign = IsSigned ? Signed : Unsigned;
  if (Sign == UnknownSignedness) {
    assert(LowerBound == 0 && UpperBound == 0);
    Sign = NewSign;

    if (IsSigned) {
      LowerBound = numeric_limits<int64_t>::min();
      UpperBound = numeric_limits<int64_t>::max();
    } else {
      LowerBound = numeric_limits<uint64_t>::min();
      UpperBound = numeric_limits<uint64_t>::max();
    }
  } else if (Sign == AnySignedness) {
    Sign = NewSign;
  } else if (Sign != NewSign) {
    Sign = InconsistentSignedness;
    // TODO: handle top case
    if (LowerBound > numeric_limits<int64_t>::max()
        || UpperBound > numeric_limits<int64_t>::max()) {
      setBottom();
    }
  }
}

template<BoundedValue::MergeType MT>
bool BoundedValue::merge(const BoundedValue &Other,
                         const DataLayout &DL,
                         Type *Int64) {
  if (Bottom)
    return false;

  if (Other.Bottom) {
    setBottom();
    return true;
  }

  if (isTop() && Other.isTop()) {
    return false;
  } else if (MT == And && isTop()) {
    LowerBound = Other.LowerBound;
    UpperBound = Other.UpperBound;
    Sign = Other.Sign;
    Negated = Other.Negated;
    return true;
  } else if (MT == And && Other.isTop()) {
    return false;
  } else if (MT == Or && isTop()) {
    return false;
  } else if (MT == Or && Other.isTop()) {
    setTop();
    return true;
  }

  if (Sign == AnySignedness || Other.Sign == AnySignedness) {
    if (Sign == AnySignedness)
      Sign = Other.Sign;
  } else {
    setSignedness(Other.isSigned());
  }

  if (Bottom)
    return true;

  // We don't handle this case for now
  if (Sign == InconsistentSignedness) {
    setBottom();
    return true;
  }

  // TODO: reimplement all of this using a simple and sane range merging
  //       approach

  Predicate LE = isSigned() ? CmpInst::ICMP_SLE : CmpInst::ICMP_ULE;
  Predicate LT = isSigned() ? CmpInst::ICMP_SLT : CmpInst::ICMP_ULT;
  Predicate GE = isSigned() ? CmpInst::ICMP_SGE : CmpInst::ICMP_UGE;
  Predicate GT = isSigned() ? CmpInst::ICMP_SGT : CmpInst::ICMP_UGT;

  auto Compare = [&Int64, &DL] (uint64_t A, Predicate P, int64_t B) {
    Constant *Compare = CE::getCompare(P, CI::get(Int64, A), CI::get(Int64, B));
    return getZExtValue(Compare, DL) != 0;
  };

  const BoundedValue *LeftmostOp = this;
  const BoundedValue *RightmostOp = &Other;

  // Check that the LB of the lefmost is <= of the rightmost LB
  if (Compare(LeftmostOp->LowerBound, GT, RightmostOp->LowerBound))
    std::swap(LeftmostOp, RightmostOp);

  // If they both start at the same point, LeftmostOp is the largest
  if (Compare(LeftmostOp->LowerBound, CmpInst::ICMP_EQ, RightmostOp->LowerBound)
      && Compare(RightmostOp->UpperBound, GT, LeftmostOp->UpperBound))
    std::swap(LeftmostOp, RightmostOp);

  enum {
    Disjoint,
    Overlapping
  } Overlap;

  bool LowerLT = Compare(LeftmostOp->LowerBound, LT, RightmostOp->LowerBound);
  bool LowerLE = Compare(LeftmostOp->LowerBound, LE, RightmostOp->LowerBound);
  bool UpperGT = Compare(LeftmostOp->UpperBound, GT, RightmostOp->UpperBound);
  bool UpperGE = Compare(LeftmostOp->UpperBound, GE, RightmostOp->UpperBound);
  bool StrictlyIncluded = LowerLT && UpperGT;
  bool Included = LowerLE && UpperGE;

  if (Compare(LeftmostOp->UpperBound, LT, RightmostOp->LowerBound))
    Overlap = Disjoint;
  else
    Overlap = Overlapping;

  const BoundedValue *NegatedOp = nullptr;
  const BoundedValue *NonNegatedOp = nullptr;
  enum {
    NoNegated,
    OneNegated,
    BothNegated
  } Operands;

  if (!Negated && !Other.Negated) {
    Operands = NoNegated;
  } else if (Negated && Other.Negated) {
    Operands = BothNegated;
  } else {
    Operands = OneNegated;
    if (Negated) {
      NegatedOp = this;
      NonNegatedOp = &Other;
    } else {
      NegatedOp = &Other;
      NonNegatedOp = this;
    }
  }

  uint64_t OldLowerBound = LowerBound;
  uint64_t OldUpperBound = UpperBound;
  bool OldNegated = Negated;

  // In the following table we report all the possible situations and the
  // relative result we produce:
  //
  // type overlap     op1 op2 result
  // ======================================
  // and  disjoint    +   +   bottom
  // and  disjoint    +   -   op1
  // and  disjoint    -   -   bottom
  // and  overlapping +   +   intersection
  // and  overlapping +   -   op1-op2
  // and  overlapping -   -   !union
  // or   disjoint    +   +   bottom
  // or   disjoint    +   -   op2
  // or   disjoint    -   -   top
  // or   overlapping +   +   union
  // or   overlapping +   -   !(op2-op1)
  // or   overlapping -   -   !intersection
  //

  bool Changed = false;
  if (MT == And) {
    switch(Overlap) {
    case Disjoint:
      switch (Operands) {
      case NoNegated:
        setBottom();
        Changed = true;
        break;
      case BothNegated:
        if (LeftmostOp->LowerBound == LeftmostOp->lowerExtreme()
            && RightmostOp->UpperBound == RightmostOp->upperExtreme()) {
          std::tie(LowerBound,
                   UpperBound) = make_pair(LeftmostOp->UpperBound,
                                           RightmostOp->LowerBound);
          LowerBound++;
          UpperBound--;
          Negated = false;

          if (!Compare(LowerBound, LE, UpperBound)) {
            LowerBound = 0;
            UpperBound = 0;
            setBottom();
          }

          break;
        } else if (LeftmostOp->UpperBound + 1 == RightmostOp->LowerBound) {
          setBound<Lower, Or>(CI::get(Int64, Other.LowerBound), DL);
          if (!Bottom)
            setBound<Upper, Or>(CI::get(Int64, Other.UpperBound), DL);
          Negated = true;
        } else {
          setBottom();
          Changed = true;
        }

        break;
      case OneNegated:
        // Assign to NotNegated
        if (this != NonNegatedOp) {
          LowerBound = Other.LowerBound;
          UpperBound = Other.UpperBound;
          Negated = Other.Negated;
        }
        break;
      }
      break;
    case Overlapping:
      switch (Operands) {
      case NoNegated:
        // Intersection
        setBound<Lower, And>(CI::get(Int64, Other.LowerBound), DL);
        if (!Bottom)
          setBound<Upper, And>(CI::get(Int64, Other.UpperBound), DL);
        Negated = false;
        break;
      case OneNegated:
        // TODO: If one of the two is strictly included go to bottom
        if (StrictlyIncluded
            || (LowerBound == Other.LowerBound
                && UpperBound == Other.UpperBound)
            || (Included && LeftmostOp == NegatedOp)) {
          setBottom();
          Changed = true;
          break;
        }

        // NonNegated - Negated
        // [5,10] - ![8,12] => NonNegated.Up = Negated.Down - 1
        // [5,10] - ![1,12] == [0,10] - ([_,0] | [13,_])
        // [5,10] - ![0,7] => NonNegated.Down = Negated.Up + 1

        // [5,10] - ![4,7]
        // [5,10] - ![5,7]
        // [5,10] - ![6,12]
        // Check if NonNegated is after Negated
        uint64_t NewLowerBound, NewUpperBound;
        if (Compare(NonNegatedOp->LowerBound, GE, NegatedOp->LowerBound)) {
          NewLowerBound = NegatedOp->UpperBound + 1;
          NewUpperBound = NonNegatedOp->UpperBound;
        } else {
          NewLowerBound = NonNegatedOp->LowerBound;
          NewUpperBound = NegatedOp->LowerBound - 1;
        }
        LowerBound = NewLowerBound;
        UpperBound = NewUpperBound;
        Negated = false;
        break;
      case BothNegated:
        // Negated union
        setBound<Lower, Or>(CI::get(Int64, Other.LowerBound), DL);
        if (!Bottom)
          setBound<Upper, Or>(CI::get(Int64, Other.UpperBound), DL);
        Negated = true;
        break;
      }
      break;
    }
  } else if (MT == Or) {
    switch(Overlap) {
    case Disjoint:
      switch (Operands) {
      case NoNegated:
        if (LeftmostOp->UpperBound + 1 == RightmostOp->LowerBound) {
          setBound<Lower, Or>(CI::get(Int64, Other.LowerBound), DL);
          if (!Bottom)
            setBound<Upper, Or>(CI::get(Int64, Other.UpperBound), DL);
        } else {
          setBottom();
          Changed = true;
        }
        break;
      case OneNegated:
        // Assign to Negated
        if (this != NegatedOp) {
          LowerBound = Other.LowerBound;
          UpperBound = Other.UpperBound;
          Negated = Other.Negated;
        }
        break;
      case BothNegated:
        setTop();
        Changed = true;
        break;
      }
      break;
    case Overlapping:
      switch (Operands) {
      case NoNegated:
        setBound<Lower, Or>(CI::get(Int64, Other.LowerBound), DL);
        if (!Bottom)
          setBound<Upper, Or>(CI::get(Int64, Other.UpperBound), DL);
        break;
      case OneNegated:
        // TODO: comment this
        if (StrictlyIncluded) {
          if (LeftmostOp == NonNegatedOp)
            setTop();
          else
            setBottom();
          Changed = true;
          break;
        }

        if ((LowerBound == Other.LowerBound
             && UpperBound == Other.UpperBound)
            || (Included && LeftmostOp == NonNegatedOp)) {
          setTop();
          Changed = true;
          break;
        }

        // ![5,25] || [6,30]
        // ![5,25] || [5,10]
        // Check if NonNegated is before Negated
        uint64_t NewLowerBound, NewUpperBound;
        if (Compare(NonNegatedOp->LowerBound, LE, NegatedOp->LowerBound)) {
          NewLowerBound = NonNegatedOp->UpperBound + 1;
          NewUpperBound = NegatedOp->UpperBound;
        } else {
          NewLowerBound = NegatedOp->LowerBound;
          NewUpperBound = NonNegatedOp->LowerBound - 1;
        }
        LowerBound = NewLowerBound;
        UpperBound = NewUpperBound;
        Negated = true;
        break;
      case BothNegated:
        setBound<Lower, And>(CI::get(Int64, Other.LowerBound), DL);
        if (!Bottom)
          setBound<Upper, And>(CI::get(Int64, Other.UpperBound), DL);
        Negated = true;
        break;
      }
      break;
    }
  }

  Changed |= (OldLowerBound != LowerBound
              || OldUpperBound != UpperBound
              || OldNegated != Negated);

  assert(Compare(LowerBound, LE, UpperBound));

  return Changed;
}

// Note: this function is implemented with lower bound restriction in mind, with
// additional changes to support bound enlargement (logical `or`) or work on the
// upper bound just set the template arguments appopriately
template<BoundedValue::Bound B,
         BoundedValue::MergeType Type>
bool BoundedValue::setBound(Constant *NewValue, const DataLayout &DL) {
  assert(Sign != UnknownSignedness && Sign != AnySignedness && !Bottom);

  uint64_t &Bound = B == Lower ? LowerBound : UpperBound;

  // Create a Constant for the current bound
  Constant *OldValue = CI::get(NewValue->getType(),
                                        Bound,
                                        isSigned());

  // If the signedness is inconsistent, check that the new value lies in the
  // signed positive area, otherwise go to bottom
  // Note: OldValue should already be in this range, thanks to `setSignedness`.
  if (Sign == InconsistentSignedness && !isPositive(NewValue, DL)) {
    setBottom();
    return true;
  }

  // Update the lower bound only if NewValue > OldValue
  Predicate CompOp = (isSigned() ?
                               CmpInst::ICMP_SGT :
                               CmpInst::ICMP_UGT);

  // If we want a logical or, flip the direction of the comparison
  if (Type == Or)
    CompOp = CmpInst::getSwappedPredicate(CompOp);

  if (B == Upper)
    CompOp = CmpInst::getSwappedPredicate(CompOp);

  // Perform the comparison and, in case, update the LowerBound
  auto *Compare = CE::getCompare(CompOp, NewValue, OldValue);
  if (getConstValue(Compare, DL)->getLimitedValue()) {
    if (isSigned())
      Bound = getSExtValue(NewValue, DL);
    else
      Bound = getZExtValue(NewValue, DL);
    return true;
  }
  return false;
}
