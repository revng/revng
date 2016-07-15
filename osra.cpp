/// \file osra.cpp
/// \brief

// Standard includes
#include <cstdint>
#include <queue>
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
#include "debug.h"
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
  assert(!isConstant());
  Constant *Min = nullptr;
  Constant *Max = nullptr;
  std::tie(Min, Max) = BV->actualBoundaries(Int64);
  Min = evaluate(Min, Int64);
  Max = evaluate(Max, Int64);

  return { Min, Max };
}

static uint64_t combineImpl(unsigned Opcode,
                            bool Signed,
                            uint64_t N,
                            IntegerType *T,
                            Constant *Op,
                            const DataLayout &DL) {
  auto *R = ConstantFoldInstOperands(Opcode, T,
                                     { CI::get(T, N, Signed), Op },
                                     DL);
  return getExtValue(R, Signed, DL);
}

bool OSR::combine(unsigned Opcode, Constant *Operand, const DataLayout &DL) {
  auto *TheType = cast<IntegerType>(Operand->getType());
  bool Multiplicative = !(Opcode == Instruction::Add
                          || Opcode == Instruction::Sub);
  bool Signed = (Opcode == Instruction::SDiv
                 || Opcode == Instruction::AShr);

  Operand = getConstValue(Operand, DL);

  uint64_t OldValue = Base;
  Base = combineImpl(Opcode, Signed, Base, TheType, Operand, DL);
  bool Changed = Base != OldValue;

  if (Multiplicative) {
    OldValue = Factor;
    Factor = combineImpl(Opcode, Signed, Factor, TheType, Operand, DL);
    Changed |= OldValue != Factor;
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
  O << Value;
  O << ", ";

  switch (Sign) {
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
    O << "*";
    break;
  }

  if (Bottom) {
    O << ", bottom";
  } else if (Sign != UnknownSignedness) {
    O << ", ";
    if (LowerBound == lowerExtreme()) {
      O << "min";
    } else {
      O << LowerBound;
    }

    O << ", ";

    if (UpperBound == upperExtreme()) {
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
    O << "  ; ";
    for (auto Constraint : ConstraintsIt->second) {
      Constraint.describe(O);
      O << " ";
    }
    O << "\n";
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

template<typename T>
class VectorSet {
public:
  void insert(T Element) {
    if (Set.find(Element) == Set.end()) {
      assert(Element->getParent() != nullptr);
      Set.insert(Element);
      Queue.push(Element);
    }
  }

  bool empty() const {
    return Queue.empty();
  }

  T pop() {
    T Result = Queue.front();
    Queue.pop();
    Set.erase(Result);
    return Result;
  }

  size_t size() const { return Queue.size(); }
private:
  std::set<T> Set;
  std::queue<T> Queue;
};

/// Given an instruction, identifies, if possible, the constant operand.  If
/// both operands are constant, it returns a Constant with the folded operation
/// and nullptr. If only one is constant, it return the constant and a reference
/// to the free operand. If none of the operands are constant returns { nullptr,
/// nullptr }. It also returns { nullptr, nullptr } if I is not commutative and
/// only the first operand is constant.
std::pair<Constant *,
          Value *> OSRAPass::identifyOperands(const Instruction *I,
                                              Type *Int64,
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
      Constants[0] = CI::get(Int64, OSRIt->second.base());
  }

  // Is the second operand constant?
  if (auto *Operand = dyn_cast<Instruction>(SecondOp)) {
    auto OSRIt = OSRs.find(Operand);
    if (OSRIt != OSRs.end() && OSRIt->second.isConstant())
      Constants[1] = CI::get(Int64, OSRIt->second.base());
  }

  // No operands are constant, or only the first one and the instruction is not
  // commutative
  if ((Constants[0] == nullptr && Constants[1] == nullptr)
      || (Constants[0] != nullptr
          && Constants[1] == nullptr
          && !I->isCommutative()))
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

  return true;
}

// Terminology:
// * OSR: Offseted Shifted Range, our main data flow value which represents the
//        result of an instruction as another value, which lies withing a
//        certain range of values, multiplied by a factor and with an
//        offset, e.g. 100 + 4 * x, with 0 < x < 4.
// * free value: a value we can't represent as an OSR of another value
// * bounded variable (or BV): a free value and the range within which it lies.
bool OSRAPass::runOnFunction(Function &F) {
  const DataLayout DL = F.getParent()->getDataLayout();
  // The Overtaken map keeps track of which load/store instructions have been
  // overtaken by another load/store, meaning that they are not "free" but can
  // be expressed in terms of another stored/loaded value
  std::map<const Value *, const Value *> Overtaken;

  auto *Int64 = Type::getInt64Ty(F.getParent()->getContext());
  using UpdateFunc = std::function<BVVector(BVVector &)>;

  std::set<BasicBlock *> BlockBlackList;
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
  OSRs.clear();
  BVs = BVMap(&BlockBlackList, &DL, Int64);
  Constraints.clear();

  // Initialize the WorkList with all the instructions in the function
  VectorSet<Instruction *> WorkList;
  auto &BBList = F.getBasicBlockList();
  for (auto &BB : make_range(BBList.begin(), BBList.end()))
    if (BlockBlackList.find(&BB) == BlockBlackList.end())
      for (auto &I : make_range(BB.begin(), BB.end()))
        WorkList.insert(&I);

  // TODO: make these member functions
  auto InBlackList = [&BlockBlackList] (BasicBlock *BB) {
    return BlockBlackList.find(BB) != BlockBlackList.end();
  };

  auto EnqueueUsers = [&BlockBlackList, &WorkList] (Instruction *I) {
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
        std::tie(ConstantOp, OtherOp) = identifyOperands(I, Int64, DL);

        if (OtherOp == nullptr) {
          if (ConstantOp != nullptr) {
            // If OtherOp is nullptr but ConstantOp is not it means we were able
            // to fold the operation in a constant
            if (!IsFree)
              OSRs.erase(I);
            OSRs.emplace(make_pair(I, OSR(getZExtValue(ConstantOp, DL))));
            EnqueueUsers(I);
          }

          // In any case, break
          break;
        }

        // Get or create an OSR for the non-constant operator, this
        // will be our starting point
        OSR NewOSR = createOSR(OtherOp, I->getParent());
        if (!IsFree && !OldOSRIt->second.isConstant()) {
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
        if (!isSupportedOperation(Opcode, ConstantOp, DL)) {
          NewOSR = OSR(&BVs.get(I->getParent(), I));
          Changed = true;
        } else {
          // Combine the base OSR with the new operation
          Changed |= NewOSR.combine(Opcode, ConstantOp, DL);
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
        auto *Comparison = cast<CmpInst>(I);
        Predicate P = Comparison->getPredicate();

        Constant *ConstOp = nullptr;
        Value *FreeOpValue = nullptr;
        Instruction *FreeOp = nullptr;
        std::tie(ConstOp, FreeOpValue) = identifyOperands(I, Int64, DL);
        if (FreeOpValue != nullptr) {
          FreeOp = dyn_cast<Instruction>(FreeOpValue);
          if (FreeOp == nullptr)
            break;
        }

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

              auto *FirstOp = dyn_cast<Instruction>(I->getOperand(0));
              if (FirstOp != nullptr) {
                auto FirstOSRIt = OSRs.find(FirstOp);
                if (FirstOSRIt != OSRs.end()) {
                  auto FirstOSR = FirstOSRIt->second;
                  if (!FirstOSR.isConstant())
                    NewConstraints.push_back(*FirstOSR.boundedValue());
                }
              }

              if (auto *SecondOp = dyn_cast<Instruction>(I->getOperand(1))) {
                auto SecondOSRIt = OSRs.find(SecondOp);
                if (SecondOSRIt != OSRs.end()) {
                  auto SecondOSR = SecondOSRIt->second;
                  if (!SecondOSRIt->second.isConstant())
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
          OSR BaseOp = createOSR(FreeOp, BB);

          if (BaseOp.boundedValue()->isBottom() || BaseOp.isRelativeTo(I))
            break;

          // Notify the BV about the sign we're going to use
          bool IsSigned = Comparison->isSigned();
          BVs.setSignedness(BB,
                            BaseOp.boundedValue()->value(),
                            IsSigned);

          // Setting the sign might lead to bottom
          if (BaseOp.boundedValue()->isBottom())
            break;

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
            NewBV.merge(BV::createEQ(NewBV.value(), NewBound, IsSigned),
                        DL, Int64);
            break;
          case CmpInst::ICMP_NE:
            NewBV.merge(BV::createNE(NewBV.value(), NewBound, IsSigned),
                        DL, Int64);
            break;
          default:
            assert(false);
            break;
          }

          NewConstraints = { NewBV };
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

          // Look for instructions using constraints that have changed
          for (auto &ConstraintUser : *Entry.Target) {
            // Avoid looking up instructions that simply cannot be there
            auto Opcode = ConstraintUser.getOpcode();
            if (Opcode != Instruction::ICmp
                && Opcode != Instruction::And
                && Opcode != Instruction::Or)
              continue;

            // Ignore instructions without an associated constraint
            auto ConstraintIt = Constraints.find(&ConstraintUser);
            if (ConstraintIt == Constraints.end())
              continue;

            // If it's using one of the changed variables, insert it in the
            // worklist
            BVVector &InstructionConstraints = ConstraintIt->second;

            bool NeedsUpdate = false;
            for (auto &Constraint : Entry.Constraints) {
              for (auto &InstructionConstraint : InstructionConstraints) {
                if (InstructionConstraint.value() == Constraint.value()) {
                  NeedsUpdate = true;
                  break;
                }
              }

              if (NeedsUpdate) {
                WorkList.insert(&ConstraintUser);
                break;
              }

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
        Value *Pointer = nullptr;
        auto TheLoad = dyn_cast<LoadInst>(I);
        auto TheStore = dyn_cast<StoreInst>(I);

        // TODO: rename SelfOSR (it's not always self)
        OSR SelfOSR;
        BVVector TheConstraints;
        bool HasConstraints = false;

        if (TheLoad != nullptr) {
          // It's a load

          // If the load doesn't have an OSR associated (or it's associated to
          // itself), propagate it forward
          if (OSRs.count(I) != 0)
            break;

          Pointer = TheLoad->getPointerOperand();
          SelfOSR = OSR(&BVs.get(I->getParent(), I));
        } else {
          // It's a store
          assert(TheStore != nullptr);
          Pointer = TheStore->getPointerOperand();
          Value *ValueOp = TheStore->getValueOperand();

          if (auto *ConstantOp = dyn_cast<Constant>(ValueOp)) {
            // We're storing a constant, create a constant OSR
            SelfOSR = OSR(getZExtValue(ConstantOp, DL));
          } else if (auto *ToStore = dyn_cast<Instruction>(ValueOp)) {
            // Compute the OSR to propagate: either the one of the value to
            // store, or a self-referencing one
            auto OSRIt = OSRs.find(ToStore);
            if (OSRIt != OSRs.end())
              SelfOSR = OSRIt->second;
            else
              SelfOSR = OSR(&BVs.get(I->getParent(), I));

            // Check if the value we're storing has a constraints
            auto ConstraintIt = Constraints.find(ToStore);
            if (ConstraintIt != Constraints.end()) {
              HasConstraints = true;
              TheConstraints = ConstraintIt->second;
            }

          }

        }

        // TODO: very important, factor the two following blocks of code, we
        //       can't handle the two propagation in parallel since OSR don't
        //       have a merge policy (and most stop on conflicts) while
        //       constraints have to be propagated and merged to all the load a
        //       certain loat or store can see.

        // Note: for simplicity, from now on comments will talk about "load
        //       instructions", however this code handles stores too.
        {
          // Initialize the work list with the instruction after the store
          std::vector<iterator_range<BasicBlock::iterator>> ExploreWL;
          ExploreWL.push_back(make_range(++I->getIterator(),
                                         I->getParent()->end()));

          // TODO: can we remove Visited?
          std::set<BasicBlock *> Visited;
          // Note: we don't insert in Visited the initial basic block, so it can
          //       get visited again to consider the part before the load
          //       instruction.

          // Conflicts contains the list of loads we're not able to overtake,
          // which we'll have to move to top (i.e. make them indepent)
          std::set<LoadInst *> Conflicts;

          while (!ExploreWL.empty()) {
            auto R = ExploreWL.back();
            ExploreWL.pop_back();

            auto *BB = R.begin()->getParent();
            assert(!BlockBlackList.count(BB));
            Visited.insert(BB);

            // Loop over the instructions from here to the end of the basic
            // block
            bool Stop = false;
            for (Instruction &Inst : R) {
              if (auto *Load = dyn_cast<LoadInst>(&Inst)) {
                // TODO: handle casts and the like
                // Is it loading from the same address of our load?
                if (Load->getPointerOperand() != Pointer)
                  continue;

                // Take the reference OSR (SelfOSR) and "contextualize" it in
                // the current BasicBlock
                OSR NewOSR = switchBlock(SelfOSR, BB);

                auto LoadOSRIt = OSRs.find(Load);
                // Check if the instruction already has an OSR
                if (LoadOSRIt != OSRs.end()) {
                  if (LoadOSRIt->second.isRelativeTo(Load)
                      || LoadOSRIt->second == NewOSR) {
                    // We already passed by here
                    Stop = true;
                  } else if (LoadOSRIt->second.isConstant()) {
                    // It's constant and different, we'll never be able to take
                    // it over
                    Stop = true;
                  } else {
                    // Obtain the value relative to which the old OSR was
                    // expressed and check if it has been overtaken by either
                    // the instruction we are propagating or the value
                    // associated to the OSR we're propagating
                    auto *BV = LoadOSRIt->second.boundedValue();
                    auto OvertakerIt = Overtaken.find(BV->value());
                    auto NewRelativeTo = NewOSR.isConstant() ?
                      nullptr : NewOSR.boundedValue()->value();
                    if (OvertakerIt != Overtaken.end()
                        && (OvertakerIt->second == NewRelativeTo
                            || OvertakerIt->second == I)) {
                      // We already overtook the load it is referring to,
                      // override safely
                      OSRs.erase(LoadOSRIt);
                    } else {
                      // We didn't overtake the load it is referring to (yet?),
                      // it's a potential conflict, register it, then stop.
                      Conflicts.insert(Load);
                      Stop = true;
                    }
                  }
                }

                if (Stop)
                  break;

                // Insert the NewOSR in OSRs and mark the load as overtaken
                OSRs.insert({ &Inst, NewOSR });

                Overtaken[Load] = I;

                // The OSR has changed, mark the load and its uses to be
                // visited again
                WorkList.insert(Load);
                EnqueueUsers(Load);

              } else if (auto *Store = dyn_cast<StoreInst>(&Inst)) {
                // Check if this store might alias the memory area we're
                // tracking
                auto *PointerOp = Store->getPointerOperand();
                if (PointerOp == Pointer
                    || (!isa<GlobalVariable>(PointerOp)
                        && !isa<AllocaInst>(PointerOp))) {
                  Stop = true;
                  break;
                }
              }

            }

            // If we didn't stop, enqueue all the non-blacklisted successors for
            // exploration
            if (!Stop) {
              for (auto *Successor : successors(BB)) {
                if (!BlockBlackList.count(Successor)
                    && !Successor->empty()
                    && !Visited.count(Successor)) {
                  ExploreWL.push_back(make_range(Successor->begin(),
                                                 Successor->end()));
                }
              }
            }

            // When we have nothing more to explore, before giving up, check all
            // the candidate conflicts to see if some of them are no longer
            // conflicts, and, if so, re-enqueue them
            if (ExploreWL.empty()) {
              for (auto It = Conflicts.begin(); It != Conflicts.end();) {
                LoadInst *Conflicting = *It;
                auto ConflictOSRIt = OSRs.find(Conflicting);
                assert(ConflictOSRIt != OSRs.end());
                auto *BV = ConflictOSRIt->second.boundedValue();
                auto OvertakerIt = Overtaken.find(BV->value());
                if (OvertakerIt != Overtaken.end()
                    && OvertakerIt->second == I) {
                  auto *ConflictingBB = Conflicting->getParent();
                  ExploreWL.push_back(make_range(Conflicting->getIterator(),
                                                 ConflictingBB->end()));
                  It = Conflicts.erase(It);
                } else
                  It++;
              }
            }

          } // End of the worklist loop

          // At this point we propagated everything we could, the remaining
          // elements in Conflicts are real conflicts, make them autonomous
          for (LoadInst *Conflict : Conflicts) {
            OSR FreeOSR = createOSR(Conflict, Conflict->getParent());
            auto ConflictOSRIt = OSRs.find(Conflict);
            assert(ConflictOSRIt != OSRs.end());
            if (ConflictOSRIt->second != FreeOSR) {
              OSRs.erase(ConflictOSRIt);
              OSRs.insert({ Conflict, FreeOSR });

              WorkList.insert(Conflict);
              EnqueueUsers(Conflict);
            }
          }
        }

        if (HasConstraints) {
          // Initialize the work list with the instruction after the store
          std::vector<iterator_range<BasicBlock::iterator>> ExploreWL;
          ExploreWL.push_back(make_range(++I->getIterator(),
                                         I->getParent()->end()));

          // TODO: can we remove Visited?
          std::set<BasicBlock *> Visited;
          // Note: we don't insert in Visited the initial basic block, so it
          //       can get visisted again to consider the part before the load
          //       instruction.

          while (!ExploreWL.empty()) {
            auto R = ExploreWL.back();
            ExploreWL.pop_back();

            auto *BB = R.begin()->getParent();
            assert(BlockBlackList.find(BB) == BlockBlackList.end());
            Visited.insert(BB);

            // Loop over the instructions from here to the end of the basic
            // block
            bool Stop = false;
            for (Instruction &Inst : R) {
              if (auto *Load = dyn_cast<LoadInst>(&Inst)) {
                // TODO: handle casts and the like
                // Is it loading from the same address of our load?
                if (Load->getPointerOperand() != Pointer)
                  continue;

                bool Changed = true;

                // Propagate the constraints
                auto LoadConstraintIt = Constraints.find(Load);
                if (LoadConstraintIt == Constraints.end()) {
                  // The load has no constraints, simply propagate the input
                  // ones
                  Constraints.insert({ &Inst, TheConstraints });
                } else {
                  // Merge the constraints (using the `or` logic) directly
                  // in-place in the load's BVVector
                  using BV = BoundedValue;
                  Changed = mergeBVVectors<BV::Or>(LoadConstraintIt->second,
                                                   TheConstraints,
                                                   DL,
                                                   Int64);
                }

                // If OSR or constraints have changed, mark the load and its
                // uses to be visited again
                if (Changed) {
                  WorkList.insert(Load);
                  EnqueueUsers(Load);
                }

              } else if (auto *Store = dyn_cast<StoreInst>(&Inst)) {
                // Check if this store might alias the memory area we're
                // tracking
                auto *PointerOp = Store->getPointerOperand();
                if (PointerOp == Pointer
                    || (!isa<GlobalVariable>(PointerOp)
                        && !isa<AllocaInst>(PointerOp))) {
                  Stop = true;
                  break;
                }
              }

            }

            // If we didn't stop, enqueue all the non-blacklisted successors for
            // exploration
            if (!Stop)
              for (auto *Successor : successors(BB))
                if (BlockBlackList.find(Successor) == BlockBlackList.end()
                    && !Successor->empty()
                    && Visited.find(Successor) == Visited.end())
                  ExploreWL.push_back(make_range(Successor->begin(),
                                                 Successor->end()));

          } // End of the worklist loop

        }

        break;
      }
    default:
      break;
    }
  }

  DBG("osr", {
      BVs.prepareDescribe();
      raw_os_ostream OutputStream(dbg);
      F.getParent()->print(OutputStream, new OSRAnnotationWriter(*this));
    });

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
        O << (BVO.first != nullptr ? BVO.first->getName() : StringRef(""));
        O << ", ";
        BVO.second.describe(O);
        O << "> || ";
      }

      O << "\n";
    }
  O << "\n";
}

std::pair<bool,
          BoundedValue&> OSRAPass::BVMap::update(BasicBlock *Target,
                                                 BasicBlock *Origin,
                                                 BoundedValue NewBV) {
  auto Index = make_pair(Target, NewBV.value());
  auto MapIt = TheMap.find(Index);
  bool Changed = true;

  MapValue *BVOVector = nullptr;

  // Have we ever seen this value for this basic block?
  if (MapIt == TheMap.end()) {
    // No, just insert it
    MapValue NewBVOVector;
    NewBVOVector.Components.push_back({ make_pair(Origin, NewBV) });
    BVOVector = &TheMap.insert({ Index, NewBVOVector }).first->second;
  } else {
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
  }

  // Re-merge all the entries
  auto &Result = summarize(Target, BVOVector);

  return { Changed, Result };
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

  setSignedness(Other.isSigned());
  if (Bottom)
    return true;

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
          std::tie(LowerBound, UpperBound) = make_pair(LeftmostOp->UpperBound,
                                                       RightmostOp->LowerBound);
          Negated = false;
          break;
        }

        setBottom();
        Changed = true;
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
        setBottom();
        Changed = true;
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
        Negated = true;
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
  assert(Sign != UnknownSignedness && !Bottom);

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
