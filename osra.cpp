/// \file osra.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <algorithm>
#include <cstdint>
#include <vector>

// LLVM includes
#include "llvm/ADT/Optional.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Pass.h"

// Boost includes
#include <boost/icl/interval_set.hpp>

// Local includes
#include "datastructures.h"
#include "debug.h"
#include "memoryaccess.h"
#include "revamb.h"
#include "ir-helpers.h"
#include "osra.h"

using namespace llvm;

using boost::icl::interval_bounds;

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

using BVVector = SmallVector<BoundedValue, 2>;

template<typename C>
static auto skip(unsigned ToSkip, C &Container)
                 -> iterator_range<decltype(Container.begin())> {
  auto Begin = std::begin(Container);
  while (ToSkip --> 0)
    Begin++;
  return make_range(Begin, std::end(Container));
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

template<BoundedValue::MergeType MT>
static bool mergeBVVectors(BVVector &Base,
                           BVVector &New,
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

class BVMap {
private:
  using MapIndex = std::pair<BasicBlock *, const Value *>;
  using BVWithOrigin = std::pair<BasicBlock *, BoundedValue>;
  struct MapValue {
    BoundedValue Summary;
    std::vector<BVWithOrigin> Components;
  };

public:
  BVMap() : BlockBlackList(nullptr), DL(nullptr), Int64(nullptr) { }

  void initialize(std::set<BasicBlock *> *BlackList,
                  const DataLayout *DL,
                  Type *Int64) {
    this->BlockBlackList = BlackList;
    this->DL = DL;
    this->Int64 = Int64;
  }

  void describe(formatted_raw_ostream &O, const BasicBlock *BB) const;

  BoundedValue &get(BasicBlock *BB, const Value *V) {
    auto Index = std::make_pair(BB, V);
    auto MapIt = TheMap.find(Index);
    if (MapIt == TheMap.end()) {
      MapValue NewBVOVector;
      NewBVOVector.Summary = BoundedValue(V);
      auto It = TheMap.insert(std::make_pair(Index, NewBVOVector)).first;
      return summarize(BB, &It->second);
    }

    MapValue &BVOs = MapIt->second;
    return BVOs.Summary;
  }

  BoundedValue *getEdge(BasicBlock *BB,
                        BasicBlock *Predecessor,
                        const Value *V) {
    auto MapIt = TheMap.find({ BB, V });
    if (MapIt != TheMap.end())
      for (auto &Component : MapIt->second.Components)
        if (Component.first == Predecessor)
          return &Component.second;

    return nullptr;
  }

  void setSignedness(BasicBlock *BB, const Value *V, bool IsSigned) {
    auto Index = std::make_pair(BB, V);
    auto MapIt = TheMap.find(Index);
    assert(MapIt != TheMap.end());

    MapValue &BVOVector = MapIt->second;
    BVOVector.Summary.setSignedness(IsSigned);
    for (BVWithOrigin &BVO : BVOVector.Components)
      BVO.second.setSignedness(IsSigned);

    summarize(BB, &MapIt->second);
  }

  /// Associate to basic block \p Target a new constraint \p NewBV coming from
  /// \p Origin
  ///
  /// \return a pair containing a boolean to indicate whether there was any
  ///         change and a reference to the updated BV
  std::pair<bool, BoundedValue &> update(BasicBlock *Target,
                                         BasicBlock *Origin,
                                         BoundedValue NewBV);

  void prepareDescribe() const {
    BBMap.clear();
    for (auto Pair : TheMap) {
      auto *BB = Pair.first.first;
      if (BBMap.find(BB) == BBMap.end())
        BBMap[BB] = std::vector<MapValue> { Pair.second };
      else
        BBMap[BB].push_back(Pair.second);
    }
  }

  BoundedValue &forceBV(Instruction *V, BoundedValue BV) {
    MapIndex I { V->getParent(), V };
    MapValue NewValue;
    NewValue.Summary = BV;
    TheMap[I] = NewValue;
    return TheMap[I].Summary;
  }

  BoundedValue &forceBV(BasicBlock *BB, Value *V, BoundedValue BV) {
    MapIndex I { BB, V };
    MapValue NewValue;
    NewValue.Summary = BV;
    TheMap[I] = NewValue;
    return TheMap[I].Summary;
  }

  void clear() {
    freeContainer(TheMap);
    freeContainer(BBMap);
  }

private:
  BoundedValue &summarize(BasicBlock *Target,
                          MapValue *BVOVectorLoopInfoWrapperPass);

  bool isForced(std::map<MapIndex, MapValue>::iterator &It) const {
    const MapIndex &Index = It->first;

    if (auto *I = dyn_cast<Instruction>(Index.second)) {
      return I->getParent() == Index.first
        && It->second.Components.size() == 0;
    } else {
      return false;
    }
  }

private:
  std::set<BasicBlock *> *BlockBlackList;
  const DataLayout *DL;
  Type *Int64;
  std::map<MapIndex, MapValue> TheMap;
  mutable std::map<const BasicBlock *, std::vector<MapValue>> BBMap;
};

class OSRA {
public:
  using UpdateFunc = std::function<BVVector(BVVector &)>;

public:
  OSRA(Function &F,
       SimplifyComparisonsPass &SCP,
       ConditionalReachedLoadsPass &RDP,
       FunctionCallIdentification &FCI,
       std::map<const Value *, const OSR> &OSRs,
       BVMap &BVs) :
    F(F),
    DL(F.getParent()->getDataLayout()),
    SCP(SCP),
    RDP(RDP),
    FCI(FCI),
    Int64(IntegerType::get(getContext(&F), 64)),
    OSRs(OSRs),
    BVs(BVs),
    PDT(true) { }

  void run();
  void dump();

  bool inBlackList(BasicBlock *BB) { return BlockBlackList.count(BB) > 0; }
  void enqueueUsers(Instruction *I);

  void propagateConstraints(Instruction *I,
                            Value *Operand,
                            UpdateFunc Updater);

  // Functions handling the various class of instructions in the DFA
  void handleArithmeticOperator(Instruction *I);
  void handleLogicalOperator(Instruction *I);
  void handleComparison(Instruction *I);
  void handleUnaryOperator(Instruction *I);
  void handleBranch(Instruction *I);
  void handleMemoryOperation(Instruction *I);

  // Helper functions employed by handleComparison

  /// \brief Given an OSR, an predicate and a constant, produce a new
  ///        BoundedValue
  ///
  /// \param BaseOp the OSR representing theleft-hand side of the comparison
  /// \param P the comparison operation to perform
  /// \param ConstOp the constant against which the comparison is performed
  ///
  /// \return a new BoundedValue constraining the BoundedValue associated to \p
  ///         BaseOp with the specified comparison
  BoundedValue mergePredicate(OSR &BaseOp, Predicate P, Constant *ConstOp);

  Optional<BoundedValue> applyConstraint(Instruction *I,
                                         OSR &BaseOp,
                                         Predicate P,
                                         Constant *ConstOp);

  std::pair<Constant *, Value *>
  identifyOperands(const Instruction *I, const DataLayout &DL) {
    return OSRAPass::identifyOperands(OSRs, I, DL);
  }

  /// \brief Possible values that an operand in a comparison can assume
  ///
  /// This data structure describe the set of possible values that the operand
  /// of a comparison can assume. They can either be constants or OSRs. If a
  /// constant is not a plain llvm::Constant but it has been obtained through a
  /// constant OSR, then we also record the Value associated to the OSR.
  ///
  /// This data structure also holds the load instruction through which we had
  /// to go through to obtain these results, and on which the comparison
  /// therefore depends.
  struct ComparisonOperand {
    ComparisonOperand() { }
    ComparisonOperand(uint64_t V) : Constants({ { V, nullptr } }) { }

    SmallVector<std::pair<uint64_t, const Value *>, 1> Constants;
    SmallVector<OSR, 4> OSRs;
    SmallVector<const LoadInst *, 3> AffectingLoads;
  };

  /// \brief Inspect a Value to produce the possible values it can assume as
  ///        comparison operand
  ///
  /// See ComparisonOperand to interpret the results.
  ComparisonOperand identifyComparisonOperands(Value *V, BasicBlock *BB) const;

  /// \brief Return true if \p I is stored in the CPU state but never read again
  bool isDead(Instruction *I) const;

  // TODO: this is a duplication of OSRAPass::getOSR
  /// \brief If available, returns the OSR associated to \p V
  const OSR *getOSR(const Value *V) const {
    auto *I = dyn_cast<Instruction>(V);

    if (I == nullptr)
      return nullptr;

    auto It = OSRs.find(I);
    if (It == OSRs.end())
      return nullptr;
    else
      return &It->second;
  }

  OSR switchBlock(OSR Base, BasicBlock *BB) const {
    Base.setBoundedValue(&BVs.get(BB, Base.boundedValue()->value()));
    return Base;
  }

  pred_iterator getValidPred(BasicBlock *BB) {
    pred_iterator Result = pred_begin(BB);
    nextValidPred(Result, pred_end(BB));
    return Result;
  }

  pred_iterator &nextValidPred(pred_iterator &It, pred_iterator End) {
    while (It != End && BlockBlackList.count(*It) != 0)
      It++;

    return It;
  }

  /// Compute a BV for \p Reached by collecting constraints on the reaching
  /// definitions over all the paths from \p Reached to them
  BoundedValue pathSensitiveMerge(LoadInst *Reached);

  bool updateLoadReacher(LoadInst *Load, Instruction *I, OSR NewOSR);
  void mergeLoadReacher(LoadInst *Load);

  /// Return a copy of the OSR associated with \p V, or if it does not exist,
  /// create a new one. In both cases the return value will refer to a bounded
  /// value in the context of \p BB.
  ///
  /// Note: after invoking this function you should always check if the result
  ///       is not expressed in terms of the instruction you're analyzing
  ///       itself, otherwise we could create (possibly infinite) loops we're
  ///       not really interested in.
  ///
  /// \return the newly created OSR, possibly expressed in terms of \p V itself.
  OSR createOSR(Value *V, BasicBlock *BB) const;

  void describe(formatted_raw_ostream &O, const Instruction *I) const;
  void describe(formatted_raw_ostream &O, const BasicBlock *BB) const;

private:
  //
  // References provided by OSRAPass
  //
  Function &F;
  const DataLayout &DL;
  SimplifyComparisonsPass &SCP;
  ConditionalReachedLoadsPass &RDP;
  FunctionCallIdentification &FCI;
  Type *Int64;

  //
  // WorkList related
  //
  std::set<BasicBlock *> BlockBlackList;
  UniquedQueue<Instruction *> WorkList;

  //
  // Data structures for the DFA
  //

  // Final information (i.e., used by OSRAPass)
  std::map<const Value *, const OSR> &OSRs;
  BVMap &BVs;

  // Temporary
  std::map<const Instruction *, BVVector> Constraints;
  using InstructionOSRVector = std::vector<std::pair<Instruction *, OSR>>;
  std::map<const LoadInst *, InstructionOSRVector> LoadReachers;

  /// Keeps track of those instruction that need to be updated when the reachers
  /// of a certain Load are updated
  using SubscribersType = SmallSet<Instruction *, 3>;
  std::map<const LoadInst *, SubscribersType> Subscriptions;

  DominatorTreeBase<BasicBlock> PDT;
};

void OSRA::propagateConstraints(Instruction *I,
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
      enqueueUsers(I);
    }
  }
}

void OSRA::handleArithmeticOperator(Instruction *I) {
  // Check if it's a free value
  auto OldOSRIt = OSRs.find(I);
  bool IsFree = OldOSRIt == OSRs.end();
  bool Changed = false;

  Constant *ConstantOp = nullptr;
  Value *OtherOp = nullptr;
  std::tie(ConstantOp, OtherOp) = identifyOperands(I, DL);

  if (OtherOp == nullptr) {
    if (ConstantOp != nullptr) {
      // If OtherOp is nullptr but ConstantOp is not it means we were able to
      // fold the operation in a constant
      if (!IsFree)
        OSRs.erase(I);

      uint64_t Constant = getZExtValue(ConstantOp, DL);
      BoundedValue ConstantBV = BoundedValue::createConstant(I, Constant);
      auto &BV = BVs.forceBV(I, ConstantBV);
      OSR ConstantOSR(&BV);
      OSRs.emplace(make_pair(I, ConstantOSR));
      enqueueUsers(I);
    }

    // In any case, return
    return;
  }

  // Get or create an OSR for the non-constant operator, this will be our
  // starting point
  OSR NewOSR = createOSR(OtherOp, I->getParent());
  if (!IsFree) {
    if (NewOSR.isRelativeTo(OldOSRIt->second.boundedValue()->value())) {
      return;
    } else {
      Changed = true;
    }
  }

  // Check we're not depending on ourselves, if we are leave us as a free value
  if (NewOSR.isRelativeTo(I)) {
    assert(IsFree);
    return;
  }

  // TODO: this is probably a bad idea
  if (NewOSR.boundedValue()->isBottom()) {
    if (!IsFree)
      OSRs.erase(OldOSRIt);
    return;
  }

  // TODO: skip this if isDead(I)
  // Update signedness information if the given operation is sign-aware
  unsigned Opcode = I->getOpcode();
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
    enqueueUsers(I);
  }
}

void OSRA::handleLogicalOperator(Instruction *I) {
  Instruction *FirstOperand = dyn_cast<Instruction>(I->getOperand(0));
  Instruction *SecondOperand = dyn_cast<Instruction>(I->getOperand(1));
  if (FirstOperand == nullptr || SecondOperand == nullptr)
    return;

  auto FirstConstraintIt = Constraints.find(FirstOperand);
  auto SecondConstraintIt = Constraints.find(SecondOperand);

  // We can merge the BVs only if both operands have one
  if (FirstConstraintIt == Constraints.end()
      || SecondConstraintIt == Constraints.end())
    return;

  // Initialize the new boundaries with the first operand
  auto NewConstraints = FirstConstraintIt->second;
  auto &OtherConstraints = SecondConstraintIt->second;

  if (I->getOpcode() == Instruction::And)
    mergeBVVectors<AndMerge>(NewConstraints, OtherConstraints, DL, Int64);
  else
    mergeBVVectors<OrMerge>(NewConstraints, OtherConstraints, DL, Int64);

  bool Changed = true;
  // If this instruction already had constraints, compare them with the new ones
  auto OldConstraintsIt = Constraints.find(I);
  if (OldConstraintsIt != Constraints.end())
    Changed = differ(OldConstraintsIt->second, NewConstraints);

  // If something changed, register the new constraints and re-enqueue all the
  // users of the instruction
  if (Changed) {
    Constraints[I] = NewConstraints;
    enqueueUsers(I);
  }
}

// TODO: give a better name
BoundedValue OSRA::mergePredicate(OSR &BaseOp, Predicate P, Constant *ConstOp) {
  const Value *V = BaseOp.boundedValue()->value();
  bool IsSigned = BaseOp.boundedValue()->isSigned();
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
    return BoundedValue::createBottom(V);

  uint64_t NewBound = getExtValue(NewBoundC, IsSigned, DL);

  // TODO: this is an hack
  if (NewBound == 0
      && (P == CmpInst::ICMP_ULT || P == CmpInst::ICMP_UGE))
    return BoundedValue(V);

  BoundedValue Constraint;
  switch (P) {
  case CmpInst::ICMP_UGT:
  case CmpInst::ICMP_UGE:
  case CmpInst::ICMP_SGT:
  case CmpInst::ICMP_SGE:
    if (CmpInst::isFalseWhenEqual(P))
      NewBound++;
    return BoundedValue::createGE(V, NewBound, IsSigned);

  case CmpInst::ICMP_ULT:
  case CmpInst::ICMP_ULE:
  case CmpInst::ICMP_SLT:
  case CmpInst::ICMP_SLE:
    if (CmpInst::isFalseWhenEqual(P))
      NewBound--;
    return BoundedValue::createLE(V, NewBound, IsSigned);

  case CmpInst::ICMP_EQ:
    return BoundedValue::createEQ(V, NewBound, IsSigned);

  case CmpInst::ICMP_NE:
    return BoundedValue::createNE(V, NewBound, IsSigned);

  default:
    llvm_unreachable("Unexpected comparison operator");
    break;
  }
}

Optional<BoundedValue> OSRA::applyConstraint(Instruction *I,
                                             OSR &BaseOp,
                                             Predicate P,
                                             Constant *ConstOp) {
  BasicBlock *BB = I->getParent();
  const BoundedValue *OriginalBV = BaseOp.boundedValue();

  // Ignore OSR that are bottom or relative to themselves
  // TODO: how can BaseOp.factor() == 0? Investigate
  if (OriginalBV->isBottom() || BaseOp.isRelativeTo(I) || BaseOp.factor() == 0)
    return Optional<BoundedValue>();

  // Notify the BV about the sign we're going to use, unless it's a comparison
  // of (in)equality
  bool IsSigned;
  if (P != CmpInst::ICMP_EQ && P != CmpInst::ICMP_NE) {
    IsSigned = ICmpInst::isSigned(P);
    BVs.setSignedness(BB, OriginalBV->value(), IsSigned);
  } else {
    // TODO: we don't know what sign to use here, so we ignore it, should we
    //       switch to AnySignedness?
    if (!OriginalBV->hasSignedness())
      return Optional<BoundedValue>();

    IsSigned = OriginalBV->isSigned();
  }

  // If setting the sign we went to bottom or still don't have it (e.g., due to
  // being top), give up
  if (OriginalBV->isBottom() || !OriginalBV->hasSignedness())
    return Optional<BoundedValue>();

  BoundedValue Result = mergePredicate(BaseOp, P, ConstOp);
  // TODO: shouldn't we push to NewConstraints a bottom BV?
  if (Result.isBottom())
    return Optional<BoundedValue>();

  // Unsigned inequations implictly say that both operands are greater than or
  // equal to zero. This means that if we have `x - 5 < 10`, we don't just know
  // that `x < 15` but also that `x - 5 >= 0`, i.e., `x >= 5`.
  if (CmpInst::isUnsigned(P)) {
    // In case NewBV is of type [x, +Inf], we temporarily turn it into [-Inf, x
    // - 1], apply the > 0, and then flip it again. In this way if we have
    // x - 3 > 30, we get NOT [4, 33], which is way more informative than
    // [33, +Inf]

    bool Flip = Result.isRightOpen();
    if (Flip)
      Result.flip();

    auto *Zero = ConstantInt::get(ConstOp->getType(), 0);
    Result.merge(mergePredicate(BaseOp, CmpInst::ICMP_UGE, Zero), DL, Int64);

    if (Flip)
      Result.flip();
  }

  if (Result.isBottom())
    return Optional<BoundedValue>();

  // Create a copy of the current value of the BV
  BoundedValue NewBV = *OriginalBV;
  NewBV.merge(Result, DL, Int64);
  return NewBV;
}

OSRA::ComparisonOperand
OSRA::identifyComparisonOperands(Value *V, BasicBlock *BB) const {
  // Is it an LLVM Constant?
  if (auto *C = dyn_cast<Constant>(V))
    return ComparisonOperand(getLimitedValue(C));

  ComparisonOperand Result;

  // Do we have an OSR for this value?
  if (auto *I = dyn_cast<Instruction>(V)) {

    OSR TheOSR = createOSR(I, BB);

    if (TheOSR.isConstant()) {
      // It's constant: register it as such
      std::pair<uint64_t, const Value *> C {
        TheOSR.constant(),
        TheOSR.boundedValue()->value()
      };
      Result.Constants.push_back(C);
    } else {
      // A normal OSR
      Result.OSRs.push_back(TheOSR);
    }

    // Now check if the OSR is referencing a Load instruction, which is
    // associated with a self-referencing OSR. In this case, add to the
    // candidate values all the reaching definitions of such load

    // Is the OSR referencing a Load?
    if (TheOSR.boundedValue() == nullptr)
      return Result;

    const Value *BaseValue = TheOSR.boundedValue()->value();

    if (BaseValue == nullptr)
      return Result;

    if (auto *Load = dyn_cast<LoadInst>(BaseValue)) {
      // Register this instruction to be visited again when Load changes
      Result.AffectingLoads.push_back(Load);

      const OSR *LoadOSR = getOSR(Load);

      // Did we get an OSR? Is it self-referencing?
      if (LoadOSR == nullptr || LoadOSR->boundedValue()->value() != Load)
        return Result;

      auto ReachersIt = LoadReachers.find(Load);

      // Does the load have at least one reacher?
      if (ReachersIt == LoadReachers.end())
        return Result;
      auto &Reachers = ReachersIt->second;
      if (Reachers.size() <= 1)
        return Result;

      for (auto &Reacher : Reachers) {
        const BoundedValue *ReacherBV = Reacher.second.boundedValue();
        // Ignore constants and self-reaching loads
        if (!Reacher.second.isConstant()
            && ReacherBV != nullptr
            && ReacherBV->value() != nullptr
            && ReacherBV->value() != Load) {
          // Note: here we don't handle constant OSR in a special way here
          Result.OSRs.push_back(Reacher.second);
        }
      }

    }

  }

  return Result;
}

void OSRA::handleComparison(Instruction *I) {
  // Ignore dead comparisons
  if (isDead(I))
    return;

  // We use data from SimplifiedComparisonAnalysis
  auto SC = SCP.getComparison(cast<CmpInst>(I));
  ICmpInst *Comparison = new ICmpInst(SC.Predicate, SC.LHS, SC.RHS);
  std::unique_ptr<ICmpInst> SimplifiedCmpInst(Comparison);

  // Collect general information
  Predicate P = Comparison->getPredicate();
  BasicBlock *BB = I->getParent();

  // First of all handle comparisons for equality (or inequality) with 0 of
  // values with one or more constraints associated (e.g., (x < 3) == 0).
  if (P == CmpInst::ICMP_EQ || P == CmpInst::ICMP_NE) {
    assert(Constraints.count(nullptr) == 0);

    bool LHSIsZero = isa<Constant>(SC.LHS) && getLimitedValue(SC.LHS) == 0;
    Instruction *LHSInst = dyn_cast<Instruction>(SC.LHS);
    bool LHSHasConstraints = Constraints.count(LHSInst) != 0;

    bool RHSIsZero = isa<Constant>(SC.RHS) && getLimitedValue(SC.RHS) == 0;
    Instruction *RHSInst = dyn_cast<Instruction>(SC.RHS);
    bool RHSHasConstraints = Constraints.count(RHSInst) != 0;

    if ((LHSIsZero && RHSHasConstraints) || (RHSIsZero && LHSHasConstraints)) {
      // If we're comparing with 0 for equality or inequality and the
      // non-constant operand has constraints, propagate them (flipped if
      // necessary).
      Value *FreeOp = LHSIsZero ? SC.RHS : SC.LHS;
      if (P == CmpInst::ICMP_EQ) {
        propagateConstraints(I, FreeOp, [] (BVVector &Constraints) {
            BVVector Result = Constraints;
            // TODO: This is wrong! !(a & b) == !a || !b, not !a && !b
            for (auto &Constraint : Result)
              Constraint.flip();
            return Result;
          });
      } else {
        propagateConstraints(I, FreeOp, [] (BVVector &Constraints) {
            return Constraints;
          });
      }

      // Do not proceed
      return;
    }
  }

  //
  // Compute a new constraint
  //

  // Check the comparison operator is supported
  switch (P) {
  case CmpInst::ICMP_UGT:
  case CmpInst::ICMP_UGE:
  case CmpInst::ICMP_SGT:
  case CmpInst::ICMP_SGE:
  case CmpInst::ICMP_ULT:
  case CmpInst::ICMP_ULE:
  case CmpInst::ICMP_SLT:
  case CmpInst::ICMP_SLE:
  case CmpInst::ICMP_EQ:
  case CmpInst::ICMP_NE:
    break;
  default:
    return;
  }

  // Sources of operands of the comparison:
  //
  // * The (simplified) comparison operands themselves
  // * Reachers of the self-referencing load of op1
  // * Reachers of the self-referencing load of op2
  //
  // Handlers:
  //
  // * no constant operands: no-op
  // * constant vs constants: if contradiction, set everything to bottom
  // * const and non-const (or viceversa): compute new constraints

  BVVector NewConstraints;
  ComparisonOperand LHS = identifyComparisonOperands(SC.LHS, BB);
  ComparisonOperand RHS = identifyComparisonOperands(SC.RHS, BB);

  // Register the current instruction to be analyzed again in case one of
  // the load it depends on changes
  for (const LoadInst *Load : LHS.AffectingLoads)
    Subscriptions[Load].insert(I);
  for (const LoadInst *Load : RHS.AffectingLoads)
    Subscriptions[Load].insert(I);

  // const vs const: either a contradiction or a tautology
  for (auto &LHSPair : LHS.Constants) {
    for (auto &RHSPair : RHS.Constants) {
      // At least one of the two operands must have a Value associated, we don't
      // handle comparison which were originally const-foldable
      Type *T = nullptr;
      if (LHSPair.second != nullptr) {
        T = LHSPair.second->getType();
      } else {
        assert(RHSPair.second != nullptr);
        T = RHSPair.second->getType();
      }

      // Check if the comparison holds. If not, set to bottom the associate
      // value
      Constant *LHSConstant = CI::get(T, LHSPair.first);
      Constant *RHSConstant = CI::get(T, RHSPair.first);
      Constant *Compare = CE::getICmp(P, LHSConstant, RHSConstant);

      // Does the comparison hold?
      if (getLimitedValue(Compare) == 0) {
        // It doens't: send everything to bottom
        if (LHSPair.second != nullptr)
          NewConstraints.push_back(BoundedValue::createBottom(LHSPair.second));
        if (RHSPair.second != nullptr)
          NewConstraints.push_back(BoundedValue::createBottom(RHSPair.second));
      }

    }
  }

  Type *T = I->getOperand(0)->getType();

  // OSR vs const
  for (auto &RHSPair : RHS.Constants) {
    Constant *ConstOp = CI::get(T, RHSPair.first);

    for (OSR &LHSOSR : LHS.OSRs) {
      OSR TheOSR = switchBlock(LHSOSR, BB);
      auto NewBV = applyConstraint(I, TheOSR, P, ConstOp);
      if (NewBV.hasValue())
        NewConstraints.push_back(*NewBV);
    }

  }

  // const vs OSR
  ICmpInst::Predicate FP = ICmpInst::getInversePredicate(P);
  for (auto &LHSPair : LHS.Constants) {
    Constant *ConstOp = CI::get(T, LHSPair.first);

    for (OSR &RHSOSR : RHS.OSRs) {
      OSR TheOSR = switchBlock(RHSOSR, BB);
      auto NewBV = applyConstraint(I, TheOSR, FP, ConstOp);
      if (NewBV.hasValue())
        NewConstraints.push_back(*NewBV);
    }

  }

  // Note: we don't handle the remaining case, i.e., OSR vs OSR

  // or-merge constraints relative to the same bounded value before propagation
  if (NewConstraints.size() > 1) {
    BVVector MergedConstraints;
    std::set<const Value *> HandledValues;

    for (auto It = NewConstraints.begin(); It != NewConstraints.end(); It++) {
      const Value *V = It->value();
      if (HandledValues.count(V) == 0) {
        HandledValues.insert(V);

        // Initialize the result with the first BV relative to V
        BoundedValue Result = *It;

        // Iterate over the remaining elements of the constraints vector
        auto ItRest = It;
        ItRest++;
        for (; ItRest != NewConstraints.end(); ItRest++) {
          // Are they relative to the same Value? If so, merge them
          if (ItRest->value() == V) {
            Result.merge<BoundedValue::Or>(*ItRest, DL, Int64);
          }
        }

        MergedConstraints.push_back(Result);
      }
    }

    NewConstraints = MergedConstraints;
  }

  // Check against the old constraints associated with this comparison
  auto OldBVsIt = Constraints.find(I);
  bool HadConstraints = OldBVsIt != Constraints.end();

  // If we had no constraints, and we still don't have them, for sure there was
  // no change
  bool Changed = !(HadConstraints && NewConstraints.size() == 0);

  // If we had constraints, check they're actually different from the new ones
  if (Changed && HadConstraints) {
    BVVector &OldBVsVector = OldBVsIt->second;
    if (NewConstraints.size() == OldBVsVector.size()) {
      bool Different = false;
      auto OldIt = OldBVsVector.begin();
      auto NewIt = NewConstraints.begin();

      // Loop over all the elements until a different one is found or we reached
      // the end
      while (!Different && OldIt != OldBVsVector.end()) {
        Different |= *OldIt != *NewIt;
        OldIt++;
        NewIt++;
      }

      Changed = Different;
    }
  }

  // If something changed replace the BV vector and re-enqueue all the users
  if (Changed) {
    Constraints[I] = NewConstraints;
    enqueueUsers(I);
  }

}

void OSRA::handleUnaryOperator(Instruction *I) {
  // Associate OSR only if the operand has an OSR and always enqueue the users
  auto *Operand = I->getOperand(0);
  OSR NewOSR = createOSR(Operand, I->getParent());
  if (NewOSR.isRelativeTo(I))
    return;

  OSRs.emplace(make_pair(I, NewOSR));
  enqueueUsers(I);

  propagateConstraints(I, Operand, [] (BVVector &BV) { return BV; });
}

void OSRA::handleBranch(Instruction *I) {
  auto *Branch = cast<BranchInst>(I);

  // Unconditional branches bring no useful information
  if (Branch->isUnconditional())
    return;

  auto *Condition = dyn_cast<Instruction>(Branch->getCondition());
  if (Condition == nullptr)
    return;

  // Were we able to handle the condition?
  auto BranchConstraintsIt = Constraints.find(Condition);
  if (BranchConstraintsIt == Constraints.end())
    return;

  // Take a reference to the constraints, and produce a complementary version
  auto &BranchConstraints = BranchConstraintsIt->second;
  BVVector FlippedBranchConstraints = BranchConstraintsIt->second;
  // TODO: This is wrong! !(a & b) == !a || !b, not !a && !b
  for (auto &BranchConstraint : FlippedBranchConstraints)
    BranchConstraint.flip();

  // Compute the set of interested basic blocks
  std::set<const BasicBlock *> AffectedSet;

  // Build worklist with all the values affected by a constraint
  OnceQueue<const Instruction *> AffectedWorkList;
  for (auto &BranchConstraint : BranchConstraints)
    if (auto *I = dyn_cast<Instruction>(BranchConstraint.value()))
      AffectedWorkList.insert(I);

  while (!AffectedWorkList.empty()) {
    const Instruction *AffectedInst = AffectedWorkList.pop();

    for (const User *U : AffectedInst->users()) {
      if (auto *I = dyn_cast<const Instruction>(U)) {
        switch (I->getOpcode()) {
        case Instruction::Add:
        case Instruction::Sub:
        case Instruction::Mul:
        case Instruction::Shl:
        case Instruction::SDiv:
        case Instruction::UDiv:
        case Instruction::LShr:
        case Instruction::AShr:
        case Instruction::ICmp:
        case Instruction::SExt:
        case Instruction::ZExt:
        case Instruction::Trunc:
        case Instruction::And:
        case Instruction::Or:
        case Instruction::Xor:
        case Instruction::Call:
        case Instruction::IntToPtr:
        case Instruction::Select:
        case Instruction::URem:
        case Instruction::SRem:
          AffectedSet.insert(I->getParent());
          AffectedWorkList.insert(I);
          break;
        case Instruction::Store:
          AffectedSet.insert(I->getParent());
          for (const LoadInst *L : RDP.getReachedLoads(I)) {
            AffectedSet.insert(L->getParent());
            AffectedWorkList.insert(L);
          }
          break;
        case Instruction::Load:
          AffectedSet.insert(I->getParent());
          // In case of load we don't need to propagate
          break;
        default:
          assert(isa<TerminatorInst>(I) && "Unexpected instruction");
          AffectedSet.insert(I->getParent());
          for (const BasicBlock *Successor : successors(I->getParent()))
            AffectedSet.insert(Successor);
          break;
        }
      }
    }

  }

  // Remove all the basic blocks post-domainated by another basic block in the
  // list
  SmallVector<const BasicBlock *, 3> RecursivelyAffected;
  for (const BasicBlock *ToCheck : AffectedSet) {
    bool Dominated = false;
    for (const BasicBlock *Other : AffectedSet) {
      if (ToCheck != Other && PDT.dominates(Other, ToCheck)) {
        Dominated = true;
        break;
      }
    }

    if (!Dominated)
      RecursivelyAffected.push_back(ToCheck);
  }

  freeContainer(AffectedSet);


  // Create and initialize the worklist with the positive constraints for the
  // true branch, and the negated constraints for the false branch
  struct WLEntry {
    WLEntry(BasicBlock *Target, BasicBlock *Origin, BVVector Constraints) :
      Target(Target), Origin(Origin), Constraints(Constraints) { }

    BasicBlock *Target;
    BasicBlock *Origin;
    BVVector Constraints;
  };

  std::vector<WLEntry> ConstraintsWL;
  if (!inBlackList(Branch->getSuccessor(0))) {
    ConstraintsWL.push_back(WLEntry(Branch->getSuccessor(0),
                                    Branch->getParent(),
                                    BranchConstraints));
  }

  if (!inBlackList(Branch->getSuccessor(1))) {
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
    SmallSet<const Value *, 5> Affected;
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
          // Check if any of the reaching definitions of this load is affected
          // by the constraints being propagated
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
              enqueueUsers(Load);
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

    // TODO: transform set into vector
    if (std::all_of(RecursivelyAffected.begin(),
                    RecursivelyAffected.end(),
                    [this, &Entry] (const BasicBlock *BB) {
                      return PDT.dominates(Entry.Target, BB);
                    }))
      continue;

    if (FCI.isCall(Entry.Origin))
      continue;

    // Propagate the new constraints to the successors (except for the
    // dispatcher)
    if (Entry.Constraints.size() != 0)
      for (BasicBlock *Successor : successors(Entry.Target))
        if (BlockBlackList.find(Successor) == BlockBlackList.end())
          ConstraintsWL.push_back(WLEntry(Successor,
                                          Entry.Target,
                                          Entry.Constraints));
  }
}

void OSRA::handleMemoryOperation(Instruction *I) {
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

      // Compute the OSR to propagate: either the one of the value to store, or
      // an OSR relative to the value being stored
      auto OSRIt = OSRs.find(ToStore);
      if (OSRIt != OSRs.end())
        SelfOSR = OSRIt->second;
      else
        SelfOSR = OSR(&BVs.get(I->getParent(), ToStore));

      // Check if the value we're storing has constraints
      auto ConstraintIt = Constraints.find(ToStore);
      HasConstraints = ConstraintIt != Constraints.end();
      if (HasConstraints)
        TheConstraints = ConstraintIt->second;

    }

  }

  auto &ReachedLoads = RDP.getReachedLoads(I);
  for (LoadInst *ReachedLoad : ReachedLoads) {
    assert(ReachedLoad != I);

    // OSR propagation first

    // Take the reference OSR (SelfOSR) and "contextualize" it in the reached
    // load's basic block
    OSR NewOSR = switchBlock(SelfOSR, ReachedLoad->getParent());

    bool Changed = updateLoadReacher(ReachedLoad, I, NewOSR);
    if (Changed)
      mergeLoadReacher(ReachedLoad);

    // Constraints propagation
    if (HasConstraints) {
      // Does the reached load carries any constraints already?
      auto ReachedLoadConstraintIt = Constraints.find(ReachedLoad);
      if (ReachedLoadConstraintIt != Constraints.end()) {
        // Merge the constraints (using the `and` logic) directly in-place in
        // the reached load's BVVector
        using BV = BoundedValue;
        Changed |= mergeBVVectors<BV::And>(ReachedLoadConstraintIt->second,
                                           TheConstraints,
                                           DL,
                                           Int64);
      } else {
        // The reached load has no constraints, simply propagate the input ones
        Constraints.insert({ ReachedLoad, TheConstraints });
        Changed = true;
      }
    }

    // If OSR or constraints have changed, mark the reached load and its uses to
    // be visited again
    if (Changed) {
      WorkList.insert(ReachedLoad);
      enqueueUsers(ReachedLoad);
      for (Instruction *Subscriber : Subscriptions[ReachedLoad])
        WorkList.insert(Subscriber);
    }

  }

}

void OSRA::enqueueUsers(Instruction *I) {
  for (User *U : I->users())
    if (auto *UI = dyn_cast<Instruction>(U))
      if (BlockBlackList.find(UI->getParent()) == BlockBlackList.end())
        WorkList.insert(UI);
}

char OSRAPass::ID = 0;

static RegisterPass<OSRAPass> X("osra", "OSRA Pass", true, true);

OSRAPass::~OSRAPass() {
  releaseMemory();
}

void OSRAPass::releaseMemory() {
  DBG("release", { dbg << "OSRAPass is releasing memory\n"; });
  freeContainer(OSRs);
  if (BVs) {
    delete BVs;
    BVs = nullptr;
  }
}

Constant *OSR::evaluate(Constant *Value, Type *Int64) const {
  Constant *BaseC = CI::get(Int64, Base, BV->isSigned());
  Constant *FactorC = CI::get(Int64, Factor, BV->isSigned());

  return CE::getAdd(BaseC, CE::getMul(FactorC, Value));
}

pair<Constant *, Constant *>
OSR::boundaries(Type *Int64, const DataLayout &DL) const {
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
  for (std::pair<uint64_t, uint64_t> &Bound : Result.Bounds) {
    if (Bound.first != Result.lowerExtreme()) {
      Bound.first = performOp(Bound.first, I::Mul, Multiplier, DL);
      Bound.first = performOp(Bound.first, I::Add, Offset, DL);
    }

    if (Bound.second != Result.upperExtreme()) {
      Bound.second = performOp(Bound.second, I::Mul, Multiplier, DL);
      Bound.second = performOp(Bound.second, I::Add, Offset, DL);
    }
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

uint64_t OSR::BoundsIterator::operator*() const {
  bool IsSigned = TheOSR.BV->isSigned();

  auto Const = [&] (uint64_t V) { return CI::get(TheType, V, IsSigned); };
  auto *RangeStart = Const(Current->first);
  auto *RangePosition = Const(Index);
  auto *Base = Const(TheOSR.Base);
  auto *Factor = Const(TheOSR.Factor);

  auto *Result = CE::getAdd(CE::getMul(CE::getAdd(RangeStart,
                                                  RangePosition),
                                       Factor),
                            Base);

  return getLimitedValue(Result);
}

class OSRAnnotationWriter : public AssemblyAnnotationWriter {
public:
  OSRAnnotationWriter(OSRA &JTFC) : JTFC(JTFC) { }

  virtual void emitInstructionAnnot(const Instruction *I,
                                    formatted_raw_ostream &Output) {
    JTFC.describe(Output, I);
  }

  virtual void emitBasicBlockStartAnnot(const BasicBlock *BB,
                                        formatted_raw_ostream &Output) {
    JTFC.describe(Output, BB);
  }

private:
  OSRA &JTFC;
};

void OSRA::run() {
  BVs.initialize(&BlockBlackList, &DL, Int64);
  PDT.recalculate(F);

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

  // Initialize the WorkList with all the instructions in the function
  auto &BBList = F.getBasicBlockList();
  for (auto &BB : make_range(BBList.begin(), BBList.end()))
    if (BlockBlackList.find(&BB) == BlockBlackList.end())
      for (auto &I : make_range(BB.begin(), BB.end()))
        WorkList.insert(&I);

  while (!WorkList.empty()) {
    Instruction *I = WorkList.pop();

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
      handleArithmeticOperator(I);
      break;
    case Instruction::ICmp:
      handleComparison(I);
      break;
    case Instruction::ZExt:
    case Instruction::Trunc:
      handleUnaryOperator(I);
      break;
    case Instruction::And:
    case Instruction::Or:
      handleLogicalOperator(I);
      break;
    case Instruction::Br:
      handleBranch(I);
      break;
    case Instruction::Store:
    case Instruction::Load:
      handleMemoryOperation(I);
      break;
    default:
      break;
    }
  }

  DBG("osr", dump());

}

void OSRA::dump() {
  BVs.prepareDescribe();
  raw_os_ostream OutputStream(dbg);
  F.getParent()->print(OutputStream, new OSRAnnotationWriter(*this));
}

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
    for (auto Bound : Bounds) {
      O << ", [";
      if (!isConstant() && Bound.first == lowerExtreme()) {
        O << "min";
      } else {
        O << Bound.first;
      }

      O << ", ";

      if (!isConstant() && Bound.second == upperExtreme()) {
        O << "max";
      } else {
        O << Bound.second;
      }

      O << "]";
    }
  }

  O << ")";
}

void OSRA::describe(formatted_raw_ostream &O, const BasicBlock *BB) const {
  BVs.describe(O, BB);
}

void OSRA::describe(formatted_raw_ostream &O, const Instruction *I) const {
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

OSR OSRA::createOSR(Value *V, BasicBlock *BB) const {
  auto OtherOSRIt = OSRs.find(V);
  if (OtherOSRIt != OSRs.end())
    return switchBlock(OtherOSRIt->second, BB);
  else
    return OSR(&BVs.get(BB, V));
}

/// Given an instruction, identifies, if possible, the constant operand.  If
/// both operands are constant, it returns a Constant with the folded operation
/// and nullptr. If only one is constant, it return the constant and a reference
/// to the free operand. If none of the operands are constant returns { nullptr,
/// nullptr }. It also returns { nullptr, nullptr } if I is not commutative and
/// only the first operand is constant.
// TODO: this only works with commutative instructions
std::pair<Constant *, Value *>
OSRAPass::identifyOperands(std::map<const Value *, const OSR> &OSRs,
                           const Instruction *I,
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

bool OSRA::updateLoadReacher(LoadInst *Load, Instruction *I, OSR NewOSR) {
  // Check if the instruction propagating the OSR is already a component of this
  // load or not
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
      if (ReacherIt->second == NewOSR
          || ReacherIt->second.boundedValue()->value() == Load) {
        return false;
      } else {
        const Value *ReacherValue = ReacherIt->second.boundedValue()->value();
        assert(!(Reachers.size() > 1
                 && ReacherValue == Load
                 && ReacherValue != NewOSR.boundedValue()->value()));
        *ReacherIt = make_pair(I, NewOSR);
        return true;
      }
    }
  }

  LoadReachers[Load].push_back({ I, NewOSR });

  return true;
}

bool OSRA::isDead(Instruction *I) const {
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
        visitSuccessors(Store, make_blacklist(BlockBlackList), Visitor);

        return !Used;
      }
    default:
      return false;
    }
  }

  return false;
}

void OSRA::mergeLoadReacher(LoadInst *Load) {
  auto &Reachers = LoadReachers[Load];
  assert(Reachers.size() > 0);

  OSRs.erase(Load);

  // TODO: implement a real merge strategy, considering input boundaries
  OSR Result = Reachers[0].second;
  for (auto P : skip(1, Reachers)) {
    OSR ReachingOSR = P.second;
    if (ReachingOSR != Result) {
      OSR FreeOSR = createOSR(Load, Load->getParent());
      if (Reachers.size() == RDP.getReachingDefinitionsCount(Load))
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
  bool isLTR(BasicBlock *BB) const { return LTR.count(BB) != 0; }

  /// \brief Register \p BB as a basic block leading to this definition
  bool registerLTR(BasicBlock *BB) { return LTR.insert(BB).second; }

  /// \brief Mark this Reacher as active at the current height
  void setActive() { LastActiveHeight = Active; }

  /// \brief Mark this Reacher as inactive at height \p Height
  void setInactive(unsigned Height) { LastActiveHeight = Height; }

  /// \brief Set the last height of the stack when a merge was performed
  void setLastMerge(unsigned Height) { LastMergeHeight = Height; }

  /// \brief Retrieve the last height of the stack when a merge was performed
  unsigned lastMerge() const { return LastMergeHeight; }

  /// Compute a BV relative to \p V by applying the OSR associated to this
  /// definition and the constraints accumulated in Summary
  BoundedValue computeBV(Value *V, const DataLayout &DL, Type *Int64) const {
    auto Result = ReachingOSR.apply(Summary, V, DL);
    if (!Result.hasSignedness())
      Result.setBottom();

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

BoundedValue OSRA::pathSensitiveMerge(LoadInst *Reached) {
  // Initialization steps
  const unsigned MaxDepth = 10;
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
        << " is " << getName(P.first)
        << " (relative to "
        << getName(P.second.boundedValue()->value()) << ")\n";);
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
    std::string Indent(Height * 2, ' ');

    DBG("psm", dbg << Indent << "Exploring " << getName(Pred) << "\n";);

    // Check if any store in Pred can alias ReachedMA
    bool MayAlias = MemoryAccess::mayAlias(Pred, ReachedMA, DL);

    // Hold whether we should proceed to the predecessors or not Initialize to
    // false, the code handling the various reacher will enable this flag if at
    // least one of the reachers is active
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
        DBG("psm", dbg << Indent << "  Merging reacher " << ReacherIndex
            << " (relative to " << getName(R.Summary.value()) << ")\n";);

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

          if (EdgeBV != nullptr) {
            // And-merge
            BoundedValue Tmp = Result;
            Tmp.merge<BoundedValue::And>(*EdgeBV, DL, Int64);

            DBG("psm", {
                dbg << Indent << "    Got ";
                EdgeBV->describe(FormattedStream);
                dbg << " from the " << getName(*ToMerge.PredecessorIt)
                    << " -> " << getName(ToMerge.BB)
                    << " edge: ";
                Tmp.describe(FormattedStream);
                dbg << "\n";
              });

            if (Tmp.isBottom())
              break;
            else
              Result = Tmp;

          } else {
            DBG("psm", {
                dbg << Indent << "    Got no info"
                    << " from the " << getName(*ToMerge.PredecessorIt)
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

          // Deactivate
          R.setInactive(Height);
        } else {
          DBG("psm", dbg << Indent
              << "    We got an incoherent situation, ignore it\n";);
        }

      } else if (MayAlias) {
        DBG("psm", dbg << Indent
            << "  Deactivating reacher " << ReacherIndex << "\n";);

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
    DBG("psm", if (!(InStack.count(Pred) == 0)) {
        dbg << Indent
            << "    It's already on the stack\n";
      });

    // Check we're not exceeding the maximum allowed depth
    Proceed &= Height < MaxDepth;
    DBG("psm", if (!(Height < MaxDepth)) {
        dbg << Indent
            << "    We exceeded the maximum depth\n";
      });

    // Check we have at least a non-dispatcher predecessor
    pred_iterator NewPredIt = getValidPred(Pred);
    Proceed &= NewPredIt != pred_end(Pred);
    DBG("psm", if (!(NewPredIt != pred_end(Pred))) {
        dbg << Indent
            << "    No predecessors\n";
      });

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

  DBG("psm", {
      unsigned I = 0;
      for (const Reacher &R : Reachers) {
        BoundedValue ReacherBV = R.computeBV(Reached, DL, Int64);
        dbg << "Reacher " << ++I << ": ";
        ReacherBV.describe(FormattedStream);
        dbg << " (from ";
        R.osr().describe(FormattedStream);
        dbg << ")\n";
      }
    });

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
//
// * OSR: Offset Shifted Range, our main data flow value which represents the
//        result of an instruction as another value, which lies withing a
//        certain range of values, multiplied by a factor and with an
//        offset, e.g. 100 + 4 * x, with 0 < x < 4.
// * free value: a value we can't represent as an OSR of another value
// * bounded variable (or BV): a free value and the range within which it lies.
bool OSRAPass::runOnFunction(Function &F) {
  DBG("passes", { dbg << "Starting OSRAPass\n"; });

  releaseMemory();
  BVs = new BVMap();

  OSRA TheOSRA(F,
               getAnalysis<SimplifyComparisonsPass>(),
               getAnalysis<ConditionalReachedLoadsPass>(),
               getAnalysis<FunctionCallIdentification>(),
               OSRs,
               *BVs);
  TheOSRA.run();

  DBG("passes", { dbg << "Ending OSRAPass\n"; });
  return false;
}

void BVMap::describe(formatted_raw_ostream &O, const BasicBlock *BB) const {
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

std::pair<bool, BoundedValue &> BVMap::update(BasicBlock *Target,
                                              BasicBlock *Origin,
                                              BoundedValue NewBV) {
  // Debug support
  raw_os_ostream OsOstream(dbg);
  formatted_raw_ostream FormattedStream(OsOstream);
  FormattedStream.SetUnbuffered();

  DBG("osr-bv", {
      dbg << "Updating " << getName(Target)
          << " from " << getName(Origin)
          << " with ";
      NewBV.describe(FormattedStream);
      dbg << ": ";
    });

  auto Index = make_pair(Target, NewBV.value());
  auto MapIt = TheMap.find(Index);
  MapValue *BVOVector = nullptr;

  // Have we ever seen this value for this basic block?
  if (MapIt == TheMap.end()) {
    DBG("osr-bv", dbg << "new\n");

    // No, just insert it
    MapValue NewBVOVector;
    NewBVOVector.Components.push_back({ make_pair(Origin, NewBV) });
    BVOVector = &TheMap.insert({ Index, NewBVOVector }).first->second;
    return { true, summarize(Target, BVOVector) };
  } else if (isForced(MapIt)) {
    DBG("osr-bv", dbg << "forced\n");

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
    if (Base == nullptr) {
      DBG("osr-bv", dbg << "new component");

      BVOVector->Components.push_back({ Origin, NewBV });
    } else {
      DBG("osr-bv", {
          dbg << "merging with ";
          Base->describe(FormattedStream);
        });

      Changed = Base->merge<AndMerge>(NewBV, *DL, Int64);

      DBG("osr-bv", {
          dbg << " producing ";
          Base->describe(FormattedStream);
        });
    }

    // Re-merge all the entries
    auto &Result = summarize(Target, BVOVector);

    DBG("osr-bv", {
        dbg << ", final result ";
        Result.describe(FormattedStream);
        dbg << "\n";
      });

    return { Changed, Result };
  }

  // TODO: should Changed be false if isForced?

}

BoundedValue &BVMap::summarize(BasicBlock *Target, MapValue *BVOVector) {

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
    assert(Bounds.size() == 0);
    Sign = NewSign;

    if (IsSigned) {
      Bounds.emplace_back(numeric_limits<int64_t>::min(),
                          numeric_limits<int64_t>::max());
    } else {
      Bounds.emplace_back(numeric_limits<uint64_t>::min(),
                          numeric_limits<uint64_t>::max());
    }
  } else if (Sign == AnySignedness) {
    Sign = NewSign;
  } else if (Sign != NewSign) {
    Sign = InconsistentSignedness;
    // TODO: handle top case
    auto Condition = [] (std::pair<uint64_t, uint64_t> P) {
      return P.first > numeric_limits<int64_t>::max()
        || P.second > numeric_limits<int64_t>::max();
    };
    if (std::any_of(Bounds.begin(), Bounds.end(), Condition))
      setBottom();
  }
}

class BoundedValueHelpers {
public:
  template<typename T>
  static boost::icl::interval_set<T> getInterval(const BoundedValue &BV) {
    assert(!BV.isBottom());

    using interval_set = boost::icl::interval_set<T>;
    using interval = boost::icl::interval<T>;

    interval_set Result;

    for (std::pair<uint64_t, uint64_t> Bound : BV.Bounds)
      Result += interval::closed(static_cast<T>(Bound.first),
                                 static_cast<T>(Bound.second));
    if (BV.Negated) {
      interval_set FullRange;
      FullRange += interval::closed(BV.lowerExtreme(), BV.upperExtreme());

      interval_set Xor = Result ^ FullRange;
      Result.clear();
      for (auto I : Xor) {
        T Lower = I.lower();
        T Upper = I.upper();

        auto Type = I.bounds().bits();
        if (Type == interval_bounds::static_open
            || Type == interval_bounds::static_left_open)
          Lower++;

        if (Type == interval_bounds::static_open
            || Type == interval_bounds::static_right_open)
          Upper--;

        Result += interval::closed(Lower, Upper);
      }
    }

    return Result;
  }

  template<typename T>
  static BoundedValue getBV(const BoundedValue &Base,
                            boost::icl::interval_set<T> Intervals) {

    BoundedValue Result(Base.value());

    if (Intervals.iterative_size() == 0) {
      Result.setBottom();
    } else {
      using BV = BoundedValue;
      Result.Sign = Base.isSigned() ? BV::Signed : BV::Unsigned;
      assert(Result.Bounds.size() == 0 || Result.Bounds.size() == 1);
      Result.Bounds.clear();
      for (auto Interval : Intervals) {
        assert(Interval.bounds().bits() == interval_bounds::static_closed);
        Result.Bounds.emplace_back(static_cast<uint64_t>(Interval.lower()),
                                   static_cast<uint64_t>(Interval.upper()));

        for (auto &Pair : Result.Bounds) {
          if (&Pair != &Result.Bounds.back()) {
            assert(Pair != Result.Bounds.back());
          }
        }

      }
      assert(Result.Bounds.size() != 0);

    }

    return Result;
  }

};

template<BoundedValue::MergeType MT, typename T>
BoundedValue BoundedValue::mergeImpl(const BoundedValue &Other) const {
  using interval_set = boost::icl::interval_set<T>;
  interval_set Result;

  Result += BoundedValueHelpers::getInterval<T>(*this);
  DBG("bv-merge", dbg << Result);
  if (MT == And) {
    DBG("bv-merge", dbg << " & " << BoundedValueHelpers::getInterval<T>(Other));
    Result &= BoundedValueHelpers::getInterval<T>(Other);
  } else {
    DBG("bv-merge", dbg << " + " << BoundedValueHelpers::getInterval<T>(Other));
    Result += BoundedValueHelpers::getInterval<T>(Other);
  }

  DBG("bv-merge", dbg << " = " << Result << "\n");
  return BoundedValueHelpers::getBV<T>(*this, Result);
}

template<BoundedValue::MergeType MT>
bool BoundedValue::merge(const BoundedValue &Other,
                         const DataLayout &DL,
                         Type *Int64) {

  if (MT == And) {
    // x & bottom = bottom
    if (Bottom)
      return false;

    if (Other.Bottom) {
      setBottom();
      return true;
    }
  } else {
    // x | bottom = x
    if (Other.Bottom)
      return false;

    if (Bottom) {
      *this = Other;
      return true;
    }
  }

  if (isTop() && Other.isTop()) {
    return false;
  } else if (MT == And && isTop()) {
    *this = Other;
    return true;
  } else if (MT == And && Other.isTop()) {
    return false;
  } else if (MT == Or && isTop()) {
    return false;
  } else if (MT == Or && Other.isTop()) {
    setTop();
    return true;
  }

  if (Sign == AnySignedness && Other.Sign == AnySignedness) {
    setBottom();
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
  if (Sign == InconsistentSignedness || Other.Sign == InconsistentSignedness) {
    setBottom();
    return true;
  }

  BoundedValue Result;
  if (isSigned())
    Result = mergeImpl<MT, int64_t>(Other);
  else
    Result = mergeImpl<MT, uint64_t>(Other);

  if (*this != Result) {
    *this = Result;
    return true;
  } else {
    return false;
  }
}

bool BoundedValue::slowCompare(const BoundedValue &Other) const {
  assert(hasSignedness() && Other.hasSignedness());
  assert(Sign == Other.Sign);

  using H = BoundedValueHelpers;
  if (isSigned())
    return H::getInterval<int64_t>(*this) == H::getInterval<int64_t>(Other);
  else
    return H::getInterval<uint64_t>(*this) == H::getInterval<uint64_t>(Other);
}

BoundedValue::BoundsVector BoundedValue::bounds() const {
  assert(hasSignedness());

  BoundsVector Result;
  using H = BoundedValueHelpers;
  if (isSigned())
    for (auto Bound : H::getInterval<int64_t>(*this))
      Result.emplace_back(static_cast<uint64_t>(Bound.lower()),
                          static_cast<uint64_t>(Bound.upper()));
  else
    for (auto Bound : H::getInterval<uint64_t>(*this))
      Result.emplace_back(static_cast<uint64_t>(Bound.lower()),
                          static_cast<uint64_t>(Bound.upper()));

  return Result;
}
