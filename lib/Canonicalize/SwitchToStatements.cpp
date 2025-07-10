//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>
#include <functional>
#include <map>
#include <memory_resource>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ABI/ModelHelpers.h"
#include "revng/ADT/GenericGraph.h"
#include "revng/ADT/SmallMap.h"
#include "revng/InitModelTypes/InitModelTypes.h"
#include "revng/LocalVariables/LocalVariableBuilder.h"
#include "revng/MFP/MFP.h"
#include "revng/MFP/SetLattices.h"
#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Support/Debug.h"
#include "revng/Support/DecompilationHelpers.h"
#include "revng/Support/FunctionTags.h"

static Logger<> Log{ "switch-to-statements" };

using namespace llvm;

template<bool IsLegacy>
using AssignType = std::conditional_t<IsLegacy, CallInst, StoreInst>;

template<bool IsLegacy>
using CopyType = std::conditional_t<IsLegacy, CallInst, LoadInst>;

template<bool IsLegacy>
using LocalVarType = std::conditional_t<IsLegacy, CallInst, AllocaInst>;

template<bool IsLegacy>
struct AvailableExpression {
  using AssignType = AssignType<IsLegacy>;

  // The expression that is available
  Instruction *Expression = nullptr;

  // The Assign/Store that has assigned the Expression to some location.
  // It can be used to retrieve the address of the location itself.
  // nullptr means that we don't have a specific address but the Expression
  // itself can be computed at the given program point without breaking
  // semantics.
  AssignType *Assignment = nullptr;

  bool operator==(const AvailableExpression &) const = default;
  std::strong_ordering operator<=>(const AvailableExpression &) const = default;
};

template<bool IsLegacy>
using AvailableSet = std::set<AvailableExpression<IsLegacy>>;

constexpr size_t SmallSize = 8;
using InstructionVector = SmallVector<Instruction *, SmallSize>;
using InstructionSetVector = SmallSetVector<Instruction *, SmallSize>;

struct ProgramPointData {
  Instruction *TheInstruction = nullptr;
  ProgramPointData(Instruction *I) : TheInstruction(I){};
};

template<bool IsLegacy>
static auto
findAvailableRange(const AvailableSet<IsLegacy> &Availables, Instruction *I) {
  using AvailableExpression = AvailableExpression<IsLegacy>;
  auto Begin = Availables.lower_bound(AvailableExpression{
    .Expression = I, .Assignment = nullptr });
  auto End = Availables.upper_bound(AvailableExpression{
    .Expression = std::next(I), .Assignment = nullptr });
  return llvm::make_range(Begin, End);
}

using ProgramPointNode = BidirectionalNode<ProgramPointData>;
using ProgramPointsCFG = GenericGraph<ProgramPointNode>;

template<bool IsLegacy>
struct AvailableExpressionsAnalysis;

template<bool IsLegacy>
using AEA = AvailableExpressionsAnalysis<IsLegacy>;

template<bool IsLegacy>
struct AvailableExpressionsAnalysis {
  using GraphType = ProgramPointsCFG *;
  using LatticeElement = AvailableSet<IsLegacy>;
  using Label = ProgramPointNode *;
  using MFPResult = MFP::MFPResult<LatticeElement>;

  LatticeElement combineValues(const LatticeElement &LHS,
                               const LatticeElement &RHS) const {
    return SetIntersectionLattice<LatticeElement>::combineValues(LHS, RHS);
  }

  bool isLessOrEqual(const LatticeElement &LHS,
                     const LatticeElement &RHS) const {
    return SetIntersectionLattice<LatticeElement>::isLessOrEqual(LHS, RHS);
  }

  LatticeElement applyTransferFunction(ProgramPointNode *L,
                                       const LatticeElement &E) const;
};

template<bool IsLegacy>
using LatticeElement = AEA<IsLegacy>::LatticeElement;

template<bool IsLegacy>
using MFPResult = AEA<IsLegacy>::MFPResult;

template<bool IsLegacy>
using ResultMap = std::map<ProgramPointNode *, MFPResult<IsLegacy>>;

template<bool IsLegacy>
static bool isStatement(const Instruction *I) {
  // TODO: this is workaround for SelectInst being often involved in nasty
  // huge dataflows.
  // In the future we should drop this from here and add a separate pass after
  // this, that takes care of forcing local variables for nasty dataflows.
  if (isa<SelectInst>(I))
    return true;

  return hasSideEffects(*I);
}

template<bool IsLegacy>
static bool isProgramPoint(const Instruction *I) {

  const Instruction *UnexpectedInstruction = nullptr;
  if constexpr (IsLegacy) {
    // Legacy mode just assumes that we don't have Load/Store/Alloca at all.
    if (isa<LoadInst>(I) or isa<StoreInst>(I) or isa<AllocaInst>(I)
        or isa<PHINode>(I))
      UnexpectedInstruction = I;
  } else {
    // Non-legacy mode assumes that most custom opcode don't exist. Some of them
    // have been replaced by Load/Store/Alloca, and others have been dropped
    // because in the clift-based pipeline they will be only materialized in
    // Clift as regular operators, so we don't need them in LLVM anymore and we
    // want to make sure they disappear over time until we can actually drop
    // them.
    if (isCallToTagged(I, FunctionTags::AllocatesLocalVariable)
        or isCallToTagged(I, FunctionTags::LocalVariable)
        or isCallToTagged(I, FunctionTags::Copy)
        or isCallToTagged(I, FunctionTags::Assign)
        or isCallToTagged(I, FunctionTags::AddressOf)
        or isCallToTagged(I, FunctionTags::Marker)
        or isCallToTagged(I, FunctionTags::IsRef)
        or isCallToTagged(I, FunctionTags::StringLiteral)
        or isCallToTagged(I, FunctionTags::ModelCast)
        or isCallToTagged(I, FunctionTags::ModelGEP)
        or isCallToTagged(I, FunctionTags::ModelGEPRef)
        or isCallToTagged(I, FunctionTags::Parentheses)
        or isCallToTagged(I, FunctionTags::LiteralPrintDecorator)
        or isCallToTagged(I, FunctionTags::HexInteger)
        or isCallToTagged(I, FunctionTags::CharInteger)
        or isCallToTagged(I, FunctionTags::BoolInteger)
        or isCallToTagged(I, FunctionTags::NullPtr)
        or isCallToTagged(I, FunctionTags::SegmentRef)
        or isCallToTagged(I, FunctionTags::UnaryMinus)
        or isCallToTagged(I, FunctionTags::BinaryNot)
        or isCallToTagged(I, FunctionTags::BooleanNot)) {
      UnexpectedInstruction = I;
    }
  }

  if (nullptr != UnexpectedInstruction) {
    I->dump();
    revng_abort("Unexpected Instruction");
  }

  return I == &I->getParent()->front() or isStatement<IsLegacy>(I)
         or mayReadMemory(*I);
}

static RecursiveCoroutine<std::optional<const Value *>>
getAccessedLocalVariableFromModelGEP(const CallInst *ModelGEPRefCall) {
  revng_assert(isCallToTagged(ModelGEPRefCall, FunctionTags::ModelGEPRef));

  revng_assert(ModelGEPRefCall->arg_size() >= 2);

  // If the ModelGEPRefCall has more than 2 arguments, and some of them are not
  // constants, we cannot figure out all the list of potentially accessed local
  // variables, so we just return nullptr.
  for (const Use &GEPArg : llvm::drop_begin(ModelGEPRefCall->args(), 2)) {
    if (not isa<Constant>(GEPArg.get()))
      rc_return nullptr;
  }

  // If the Base argument of the ModelGEPRefCall isn't a LocalVariable, nor an
  // Argument, nor another ModelGEPRef, we just return nullopt, meaning that
  // this thing doesn't really access any local variable.
  auto *GEPBase = ModelGEPRefCall->getArgOperand(1);
  // If the GEPBase is directly an argument, we're done
  if (isa<Argument>(GEPBase))
    rc_return GEPBase;

  // If the GEPBase is directly a LocalVariable, we're done
  if (isCallToTagged(GEPBase, FunctionTags::AllocatesLocalVariable))
    rc_return GEPBase;

  // If the GEPBase is another ModelGEPRef we recur.
  // Notice that we don't recur on ModelGEP, only on ModelGEPRef, because simple
  // ModelGEP can have arbitrary base pointers, but they never access
  // LocalVariables.
  if (auto *NestedModelGEPRef = getCallToTagged(GEPBase,
                                                FunctionTags::ModelGEPRef))
    rc_return rc_recur getAccessedLocalVariableFromModelGEP(NestedModelGEPRef);

  // Everything else cannot access local variables, so we return nullopt.
  rc_return std::nullopt;
}

template<bool IsLegacy>
static const CopyType<IsLegacy> *getCopy(const Instruction *I) {
  if constexpr (IsLegacy)
    return getCallToTagged(I, FunctionTags::Copy);
  else
    return dyn_cast<LoadInst>(I);
}

template<bool IsLegacy>
static const AssignType<IsLegacy> *getAssign(const Instruction *I) {
  if constexpr (IsLegacy)
    return getCallToTagged(I, FunctionTags::Assign);
  else
    return dyn_cast<StoreInst>(I);
}

// Get the local variable accessed by I.
// If the returned optional is nullopt, it means that I doesn't accessy memory.
// If the returned optional is engaged:
// - if it holds a null pointer, it means that I accesses memory but we weren't
//   able to figure out where
// - if it holds a valid pointer, it must be either an Argument or the accessed
//   local variable
template<bool IsLegacy>
static std::optional<const Value *> getAccessedLocal(const Instruction *I) {

  const CopyType<IsLegacy> *Copy = getCopy<IsLegacy>(I);
  const AssignType<IsLegacy> *Assign = getAssign<IsLegacy>(I);

  // If it's not a Copy not an Assign then it's not an access to a local
  // variable.
  // TODO: this doesn't take into consideration stuff like memcpy.
  if (not Copy and not Assign)
    return std::nullopt;

  if constexpr (IsLegacy) {
    const CallInst *AccessCall = Copy ? Copy : Assign;

    unsigned AccessArgumentNumber = Assign ? 1 : 0;
    const auto *Accessed = AccessCall->getArgOperand(AccessArgumentNumber);

    // If the accessed thing is directly an Argument or a LocalVariable we're
    // done.
    if (isa<Argument>(Accessed)
        or isCallToTagged(Accessed, FunctionTags::AllocatesLocalVariable)) {
      return Accessed;
    }

    // If the accessed thing is not a ModelGEPRef, then it's not an access to a
    // local variable.
    auto *ModelGEPRef = getCallToTagged(Accessed, FunctionTags::ModelGEPRef);
    if (not ModelGEPRef)
      return std::nullopt;

    return getAccessedLocalVariableFromModelGEP(ModelGEPRef);
  } else {

    unsigned PointerOperandNo = Copy ? LoadInst::getPointerOperandIndex() :
                                       StoreInst::getPointerOperandIndex();
    const Value *PointerOperand = I->getOperand(PointerOperandNo);
    // If the pointer operand of the access is an argument return it.
    if (isa<Argument>(PointerOperand))
      return PointerOperand;

    // If the pointer operand of the access is an AllocaInst
    // representing a local variable, return it.
    const auto *AccessedLocalVariable = dyn_cast<AllocaInst>(PointerOperand);
    if (AccessedLocalVariable)
      return AccessedLocalVariable;

    // TODO: in all the other cases we would have to resort to LLVM's alias
    // analysis for providing a sensible answer.
    // For now we're not doing that, so we just return nullptr, which means I
    // accesses a local variable but we can't say which one.
    return nullptr;
  }
}

template<bool IsLegacy>
static bool localVariablesNoAlias(const Instruction *I, const Instruction *J) {

  // Copies from local variables never alias anyone else, except other
  // instructions that copy or assign the same local variable
  std::optional<const Value *> MayBeAccessedByI = getAccessedLocal<IsLegacy>(I);
  std::optional<const Value *> MayBeAccessedByJ = getAccessedLocal<IsLegacy>(J);

  // If either doesn't access a local variable, they are noAlias.
  if (not MayBeAccessedByI.has_value() or not MayBeAccessedByJ.has_value())
    return true;

  const Value *AccessedByI = *MayBeAccessedByI;
  const Value *AccessedByJ = *MayBeAccessedByJ;

  // If either is nullptr, there is at least one among I and J that access many
  // variables, and we just can't say with certainty that they are noAlias
  if (nullptr == AccessedByI or nullptr == AccessedByJ)
    return false;

  // If both are arguments, they may point overlapping memory, and we have no
  // way of knowing. So we return false, because we're not sure they don't
  // alias.
  if (isa<Argument>(AccessedByI) and isa<Argument>(AccessedByJ))
    return false;

  // For all the other cases they are noAlias only if the accessed
  // local variable is different.
  return AccessedByI != AccessedByJ;
}

template<bool IsLegacy>
static bool noAlias(const Instruction *I, const Instruction *J) {
  revng_log(Log, "noAlias?");
  LoggerIndent X{ Log };
  revng_log(Log, "I: " << dumpToString(I));
  revng_log(Log, "J: " << dumpToString(J));
  LoggerIndent XX{ Log };
  // If either instruction doesn't access memory, they are noAlias for sure.
  if (not I->mayReadOrWriteMemory() or not J->mayReadFromMemory()) {
    revng_log(Log, "I or J doesNotAccessMemory");
    return true;
  }

  // Here both instructions access memory.

  // First, handle LocalVariables specifically.
  // TODO: this is a poor's man alias analysis, which only explicitly handles
  // stuff that is frequent and that we care about. In the future we have plans
  // to replace it with a full fledged AliasAnalysis from LLVM
  if (localVariablesNoAlias<IsLegacy>(I, J)) {
    revng_log(Log, "I and J both access local variables that do not alias");
    return true;
  }

  // TODO: In all the other cases, to reason accurately about aliasing, we would
  // need LLVM's alias analysis. At the moment this is out of scope, so we
  // always fall back to false, meaning that we can't say for sure that I and J
  // do not alias.
  revng_log(Log, "I and J aren't provably noAlias");
  return false;
}

template<bool IsLegacy>
static void applyTransferFunction(Instruction *I, LatticeElement<IsLegacy> &E) {
  using AvailableExpression = AvailableExpression<IsLegacy>;
  using AssignType = AssignType<IsLegacy>;

  revng_log(Log, "applyTransferFunction on Instruction: " << dumpToString(I));
  LoggerIndent X{ Log };

  if constexpr (IsLegacy) {
    revng_assert(not isa<LoadInst>(I) and not isa<StoreInst>(I));
  } else {
    revng_assert(not isCallToTagged(I, FunctionTags::Copy)
                 and not isCallToTagged(I, FunctionTags::Assign));
  }

  if (isStatement<IsLegacy>(I)) {
    revng_log(Log, "isStatement");
    LoggerIndent XX{ Log };
    for (const AvailableExpression &A : llvm::make_early_inc_range(E)) {
      const auto &[Available, Assign] = A;
      revng_log(Log, "Available: " << dumpToString(Available));
      revng_log(Log, "Assign: " << dumpToString(Assign));
      LoggerIndent XXX{ Log };
      if (not noAlias<IsLegacy>(I, Available)) {
        revng_log(Log, "Available: " << dumpToString(Available));
        revng_log(Log, "is not noAlias (MayAlias) with I");
        revng_log(Log, "erase Available");
        E.erase(A);
      } else if (Assign and not noAlias<IsLegacy>(I, Assign)) {
        revng_log(Log, "Assign: " << dumpToString(Assign));
        revng_log(Log, "erase Available");
        E.erase(A);
      } else {
        revng_log(Log, "is noAlias with I");
      }
    }
  }

  Instruction *AssignedValue = nullptr;
  if constexpr (IsLegacy) {
    // In legacy mode assignments are calls to custom opcode Assign function
    if (auto *Assign = getCallToTagged(I, FunctionTags::Assign)) {
      AssignedValue = dyn_cast<Instruction>(Assign->getArgOperand(0));
    }
  } else {
    // In non-legacy mode assignments are StoreInst
    if (auto *Store = dyn_cast<StoreInst>(I)) {
      AssignedValue = dyn_cast<Instruction>(Store->getValueOperand());
    }
  }

  if (AssignedValue) {
    revng_log(Log, "I is Assign");
    revng_log(Log, "insert Available: " << dumpToString(AssignedValue));
    revng_log(Log, "       Assign: " << dumpToString(I));

    E.insert(AvailableExpression{
      .Expression = AssignedValue,
      .Assignment = cast<AssignType>(I),
    });
  }

  if (mayReadMemory(*I)) {
    revng_log(Log, "mayReadMemory -> insert Available: I");
    E.insert(AvailableExpression{
      .Expression = I,
      .Assignment = nullptr,
    });
  }
}

static void applyTransferFunction(Instruction *I, LatticeElement<true> &E) {
  return applyTransferFunction<true>(I, E);
}

static void applyTransferFunction(Instruction *I, LatticeElement<false> &E) {
  return applyTransferFunction<false>(I, E);
}

template<bool IsLegacy>
AEA<IsLegacy>::LatticeElement
AEA<IsLegacy>::applyTransferFunction(ProgramPointNode *ProgramPoint,
                                     const AEA<IsLegacy>::LatticeElement &E)
  const {

  Instruction *I = ProgramPoint->TheInstruction;

  revng_log(Log, "applyTransferFunction on ProgramPoint: " << dumpToString(I));
  LoggerIndent Indent{ Log };

  LatticeElement Result = E;

  revng_log(Log, "initial set");
  if (Log.isEnabled()) {
    LoggerIndent X{ Log };
    for (const auto &[Available, Assign] : Result) {
      revng_log(Log, "Available: " << dumpToString(Available));
      revng_log(Log, "Assign: " << dumpToString(Assign));
    }
  }

  ::applyTransferFunction(I, Result);

  revng_log(Log, "final set");
  if (Log.isEnabled()) {
    LoggerIndent X{ Log };
    for (const auto &[Available, Assign] : Result) {
      revng_log(Log, "Available: " << dumpToString(Available));
      revng_log(Log, "Assign: " << dumpToString(Assign));
    }
  }

  return Result;
}

template<bool IsLegacy>
static InstructionSetVector getProgramPoints(BasicBlock &B) {
  InstructionSetVector Results;
  for (Instruction &I : B)
    if (isProgramPoint<IsLegacy>(&I))
      Results.insert(&I);
  return Results;
}

using InstructionProgramPoint = std::unordered_map<const Instruction *,
                                                   ProgramPointNode *>;

// An extended version of ProgramPointsCFG, that holds a graph of statements
// points, along with a map from each Instruction to its previous statement.
template<bool IsLegacy>
class ProgramPointsGraphWithInstructionMap {
public:
  using AvailableExpression = AvailableExpression<IsLegacy>;
  using AvailableSet = AvailableSet<IsLegacy>;
  using ResultMap = ResultMap<IsLegacy>;

public:
  ProgramPointsCFG ProgramPointsGraph;

private:
  // Map an Instruction to its associated program point in ProgramPointsGraph
  InstructionProgramPoint ProgramPoint;

  // Map an Instruction to its associated previous program point in
  // ProgramPointsGraph.
  InstructionProgramPoint PreviousProgramPointInBlock;

  // Map an Instruction to its associated next program point in
  // ProgramPointsGraph.
  InstructionProgramPoint NextProgramPointInBlock;

public:
  // Factory from llvm::Function
  static ProgramPointsGraphWithInstructionMap makeFromFunction(Function &F) {

    SmallMap<BasicBlock *, std::pair<ProgramPointNode *, ProgramPointNode *>, 8>
      BlockToBeginEndNode;

    ProgramPointsGraphWithInstructionMap Result;

    ProgramPointsCFG &TheCFG = Result.ProgramPointsGraph;
    InstructionProgramPoint &ProgramPoint = Result.ProgramPoint;
    InstructionProgramPoint
      &PreviousProgramPointInBlock = Result.PreviousProgramPointInBlock;
    InstructionProgramPoint
      &NextProgramPointInBlock = Result.NextProgramPointInBlock;

    const auto MakeCFGNode = [&TheCFG, &ProgramPoint](Instruction *I) {
      ProgramPointNode *NewNode = TheCFG.addNode(I);
      ProgramPoint[I] = NewNode;
      return NewNode;
    };

    for (BasicBlock &BB : F) {
      InstructionSetVector ProgramPoints = getProgramPoints<IsLegacy>(BB);

      // Reserve space for the new ProgramPoints. This is for performance but
      // also for stability of pointers while adding new nodes, which allows to
      // also save pointers to begin and end nodes of each block in a map, to
      // handle addition of inter-block edges. If we don't reserve the pointers
      // returned by addNode aren't stable and the trick for adding inter-block
      // edges doesn't work.
      TheCFG.reserve(TheCFG.size() + ProgramPoints.size());

      ProgramPointNode *FirstNode = MakeCFGNode(ProgramPoints.front());
      ProgramPointNode *LastNode = FirstNode;
      for (Instruction &I :
           llvm::make_range(BB.begin(), ProgramPoints.front()->getIterator()))
        NextProgramPointInBlock[&I] = LastNode;

      auto ProgramPointPairs = llvm::zip_equal(llvm::drop_end(ProgramPoints),
                                               llvm::drop_begin(ProgramPoints));
      for (const auto &[PreviousProgramPoint, NextProgramPoint] :
           ProgramPointPairs) {
        // Create a new node.
        ProgramPointNode *NewNode = MakeCFGNode(NextProgramPoint);
        // We can already add intra-block edges.
        LastNode->addSuccessor(NewNode);

        // Now we have to initialize PreviousProgramPointInBlock for all the
        // instructions that are not program points and that are among the
        // previous program point and the current new one.
        for (Instruction &I :
             llvm::make_range(std::next(PreviousProgramPoint->getIterator()),
                              NextProgramPoint->getIterator()))
          PreviousProgramPointInBlock[&I] = LastNode;

        // Finally we can update the LastNode.
        LastNode = NewNode;
      }
      for (Instruction &I :
           llvm::make_range(std::next(ProgramPoints.back()->getIterator()),
                            BB.end()))
        PreviousProgramPointInBlock[&I] = LastNode;

      BlockToBeginEndNode[&BB] = { FirstNode, LastNode };
    }

    // Now we add the inter-block edges.
    for (BasicBlock &BB : F)
      for (BasicBlock *Successor : llvm::successors(&BB))
        BlockToBeginEndNode.at(&BB)
          .second->addSuccessor(BlockToBeginEndNode.at(Successor).first);

    // And set the entry node, which makes the MFP later more efficient, because
    // it allows the algorithm to take the structure of the graph into account,
    // instead of iterating in sparse order.
    TheCFG.setEntryNode(BlockToBeginEndNode.at(&F.getEntryBlock()).first);

    return Result;
  }

public:
  auto getAvailableAt(Instruction *I,
                      const Instruction *Where,
                      const ResultMap &MFPResultMap) const {

    revng_log(Log, "IsAvailableAt");
    revng_log(Log, "I: " << dumpToString(I));
    revng_log(Log, "Where: " << dumpToString(Where));

    auto ProgramPointIt = ProgramPoint.find(Where);
    if (ProgramPointIt != ProgramPoint.end()) {
      revng_log(Log, "is ProgramPoint");

      ProgramPointNode *UserProgramPoint = ProgramPointIt->second;
      const AvailableSet &Available = MFPResultMap.at(UserProgramPoint).InValue;
      return findAvailableRange(Available, I);
    }

    auto PreviousPointIt = PreviousProgramPointInBlock.find(Where);
    if (PreviousPointIt != PreviousProgramPointInBlock.end()) {
      revng_log(Log, "is NOT ProgramPoint");

      ProgramPointNode *UserProgramPoint = PreviousPointIt->second;
      revng_log(Log,
                "Previous ProgramPoint: "
                  << dumpToString(UserProgramPoint->TheInstruction));
      const AvailableSet &Available = MFPResultMap.at(UserProgramPoint)
                                        .OutValue;
      return findAvailableRange(Available, I);
    }

    auto NextProgramPointIt = NextProgramPointInBlock.find(Where);
    if (NextProgramPointIt != NextProgramPointInBlock.end()) {
      revng_log(Log, "is before first ProgramPoint in BasicBlock");

      ProgramPointNode *UserProgramPoint = NextProgramPointIt->second;
      revng_log(Log,
                "first ProgramPoint in BasicBlock: "
                  << dumpToString(UserProgramPoint->TheInstruction));
      const AvailableSet &Available = MFPResultMap.at(UserProgramPoint).InValue;
      return findAvailableRange(Available, I);
    }

    revng_abort();
  }

  bool isAvailableAt(Instruction *I,
                     const Instruction *Where,
                     const ResultMap &MFPResultMap) const {
    bool Result = not getAvailableAt(I, Where, MFPResultMap).empty();
    revng_log(Log, "Result: " << Result);
    return Result;
  }
};

template<bool IsLegacy>
using PPGWithInstructionMap = ProgramPointsGraphWithInstructionMap<IsLegacy>;

template<bool IsLegacy>
static ResultMap<IsLegacy> getMFP(ProgramPointsCFG *TheGraph) {
  using AvailableExpression = AvailableExpression<IsLegacy>;
  using AvailableSet = AvailableSet<IsLegacy>;
  using AEA = AEA<IsLegacy>;
  using AssignType = AssignType<IsLegacy>;

  AvailableSet Bottom;
  for (ProgramPointNode *N : llvm::nodes(TheGraph)) {
    Instruction *I = N->TheInstruction;

    if (mayReadMemory(*I)) {
      Bottom.insert(AvailableExpression{
        .Expression = I,
        .Assignment = nullptr,
      });
    }

    Instruction *AssignedValue = nullptr;
    if constexpr (IsLegacy) {
      // In legacy mode assignments are calls to custom opcode Assign function
      if (auto *Assign = getCallToTagged(I, FunctionTags::Assign)) {
        AssignedValue = dyn_cast<Instruction>(Assign->getArgOperand(0));
      }
    } else {
      // In non-legacy mode assignments are StoreInst
      if (auto *Store = dyn_cast<StoreInst>(I)) {
        AssignedValue = dyn_cast<Instruction>(Store->getValueOperand());
      }
    }

    if (AssignedValue) {
      Bottom.insert(AvailableExpression{
        .Expression = AssignedValue,
        .Assignment = cast<AssignType>(I),
      });
    }
  }

  AvailableSet Empty{};
  return MFP::getMaximalFixedPoint<AEA>({},
                                        TheGraph,
                                        Bottom,
                                        Empty,
                                        { TheGraph->getEntryNode() });
}

template<bool IsLegacy>
struct PickedInstructions {
  SetVector<Instruction *> ToSerialize = {};
  MapVector<Use *, AssignType<IsLegacy> *> ToReplaceWithAvailable = {};
  SmallPtrSet<AssignType<IsLegacy> *, 8> AssignToRemove = {};
};

template<bool IsLegacy>
class InstructionToSerializePicker {
public:
  using PickedInstructions = PickedInstructions<IsLegacy>;
  using PPGWithInstructionMap = PPGWithInstructionMap<IsLegacy>;
  using ResultMap = ResultMap<IsLegacy>;
  using AvailableExpression = AvailableExpression<IsLegacy>;
  using AssignType = AssignType<IsLegacy>;

public:
  InstructionToSerializePicker(Function &TheF,
                               const PPGWithInstructionMap &TheGraph,
                               const ResultMap &TheMFPResult) :
    F(TheF), Graph(TheGraph), MFPResultMap(TheMFPResult), Picked() {}

public:
  const PickedInstructions &pick() {
    Picked = {};

    // Visit in RPO for determinism
    const auto RPO = llvm::ReversePostOrderTraversal(&F);

    // First, pick all the statements amenable for serialization
    // Also compute the program order of instructions.
    size_t NextOrder = 0;
    for (BasicBlock *BB : RPO) {
      for (Instruction &I : *BB) {
        if (isStatement<IsLegacy>(&I) and not I.getType()->isVoidTy()
            and not I.getType()->isAggregateType()) {
          revng_log(Log, "I: " << dumpToString(I));
          revng_log(Log, "Picked.ToSerialize.insert(I)");
          Picked.ToSerialize.insert(&I);
        }
        ProgramOrdering[&I] = NextOrder++;
      }
    }

    // Then, start from memory reads, and traverse the dataflow to pick other
    // instructions that need to be serialized.
    for (BasicBlock *BB : RPO)
      for (Instruction &I : *BB)
        if (mayReadMemory(I))
          pickFrom(&I, &I);

    return Picked;
  }

private:
  RecursiveCoroutine<bool>
  shouldSerializeReadBeforeOrAtI(Instruction *I, Instruction *MemoryRead) {
    revng_log(Log, "PickFrom I: " << dumpToString(I));
    revng_log(Log, "MemoryRead: " << dumpToString(MemoryRead));

    LoggerIndent Indent{ Log };

    auto *IType = I->getType();

    // If I has already been picked for serialization it means that I shouldn't
    // be serialied for it.
    if (Picked.ToSerialize.contains(I)) {
      revng_log(Log, "Picked.ToSerialize.contains(I)");
      rc_return false;
    }

    // If it's a statement we must have already picked it. Just return false.
    if (isStatement<IsLegacy>(I)) {
      revng_log(Log, "I isStatement");
      if (not IType->isVoidTy() and not IType->isAggregateType()) {
        revng_assert(Picked.ToSerialize.contains(I));
      }
      rc_return false;
    }

    // If I has no uses, we are done, and there's no reason to require the
    // serialization of MemoryRead before I.
    if (not I->getNumUses()) {
      revng_log(Log, "I has no uses");
      rc_return false;
    }

    // If it exists a use U of I for which MemoryRead is not available, then
    // MemoryRead should be serialized before or at I, unless the whole
    // expression represented by I is available somewhere else.
    revng_log(Log, "Check users");
    LoggerIndent UserIndent{ Log };

    const auto IsMemoryReadAvailableAt = [this, MemoryRead](const Use &TheUse) {
      const auto *UserInstruction = cast<Instruction>(TheUse.getUser());
      return Graph.isAvailableAt(MemoryRead, UserInstruction, MFPResultMap);
    };

    const auto SerializeI =
      [I, IType = I->getType(), &ToSerialize = Picked.ToSerialize]() {
        if (not IType->isVoidTy() and not IType->isAggregateType()
            and not isCallToTagged(I, FunctionTags::IsRef)) {
          revng_log(Log,
                    "Picked.ToSerialize.serialize(I), with I: "
                      << dumpToString(I));
          ToSerialize.insert(I);
          return false;
        }
        return true;
      };

    MapVector<Use *, AssignType *> ToReplaceWithAvailable;
    SmallPtrSet<AssignType *, 8> AssignToRemove;

    // For each U Use of I where MemoryRead is not available, check if the
    // whole expression represented by I is available at U. If so add it to
    // the ToReplaceWithAvailable.
    // Otherwise if we find even a single use of I where MemoryRead is not
    // available and such that I itself is not available, we have to require I
    // to be serialized in a new local variable.
    for (Use &U : I->uses()) {
      auto *UserInstruction = cast<Instruction>(U.getUser());
      revng_log(Log, "User: " << dumpToString(UserInstruction));
      LoggerIndent MoreUserIndent{ Log };

      if (IsMemoryReadAvailableAt(U)) {
        revng_log(Log, "IsMemoryReadAvailableAt(User)");
        continue;
      }
      revng_log(Log, "MemoryRead is not available in User");

      AssignType *UserAssignCall = nullptr;
      if constexpr (IsLegacy)
        UserAssignCall = getCallToTagged(UserInstruction, FunctionTags::Assign);
      else
        UserAssignCall = dyn_cast<StoreInst>(UserInstruction);

      if (UserAssignCall) {
        // Skip over the Assign operand representing variables that are being
        // assigned, because we need to preserve them.
        if constexpr (IsLegacy) {
          if (UserAssignCall->isArgOperand(&U)
              and UserAssignCall->getArgOperandNo(&U) == 1) {
            continue;
          }
        } else {
          if (UserAssignCall == U.getUser()
              and U.getOperandNo() == StoreInst::getPointerOperandIndex())
            continue;
        }
      }

      auto AvailableRange = Graph.getAvailableAt(I,
                                                 UserInstruction,
                                                 MFPResultMap);
      if (AvailableRange.empty()) {
        revng_log(Log, "Found unavailable use. Serialize I");
        rc_return SerializeI();
      } else {
        revng_log(Log, "But I is available");

        if (auto It = Picked.ToReplaceWithAvailable.find(&U);
            It != Picked.ToReplaceWithAvailable.end()) {
          revng_log(Log,
                    "I can be read from address: " << dumpToString(It->second));

        } else {
          revng_log(Log, "Select first viable address in program order");

          AssignType *Selected = nullptr;
          size_t ProgramOrder = std::numeric_limits<size_t>::max();
          for (const AvailableExpression &A : AvailableRange) {
            if (nullptr == A.Assignment)
              continue;

            size_t NewProgramOrder = ProgramOrdering.at(A.Assignment);
            if (NewProgramOrder < ProgramOrder) {
              ProgramOrder = NewProgramOrder;
              Selected = A.Assignment;
            }
          }

          revng_log(Log, "Selected: " << dumpToString(Selected));
          if (not Selected) {
            revng_log(Log, "Selected is not an assignment. Serialize I");
            rc_return SerializeI();
          }

          if constexpr (IsLegacy)
            revng_assert(isCallToTagged(Selected, FunctionTags::Assign));
          else
            revng_assert(isa<StoreInst>(Selected));

          if (not UserAssignCall) {
            ToReplaceWithAvailable[&U] = Selected;
            continue;
          }

          std::optional<const Value *>
            MayBeAccessedByUser = getAccessedLocal<IsLegacy>(UserAssignCall);
          std::optional<const Value *>
            MayBeAccessedBySelected = getAccessedLocal<IsLegacy>(Selected);

          // If either doesn't access a local variable, we have to read from
          // there.
          if (not MayBeAccessedByUser.has_value()
              or not MayBeAccessedBySelected.has_value()) {
            ToReplaceWithAvailable[&U] = Selected;
            continue;
          }

          const Value *AccessedByUser = *MayBeAccessedByUser;
          const Value *AccessedBySelected = *MayBeAccessedBySelected;

          // If either is nullptr, there is at least one among I and J that
          // access many variables, so it's definitely not a single one and
          // cannot be optimized away.
          if (nullptr == AccessedByUser or nullptr == AccessedBySelected) {
            ToReplaceWithAvailable[&U] = Selected;
            continue;
          }

          // For all the other cases they are noAlias only if the accessed
          // local variable is different.
          if (AccessedByUser != AccessedBySelected)
            ToReplaceWithAvailable[&U] = Selected;
          else
            AssignToRemove.insert(UserAssignCall);
        }
      }
    }

    // If we reach this point, it means that no user forced us to serialize I.
    // At this point we can commit ToReplaceWithAvailable into
    // Picked.ToReplaceWithAvailable.
    for (const auto &Element : ToReplaceWithAvailable)
      Picked.ToReplaceWithAvailable.insert(Element);
    // And we can also commit the fact that we want to remove the Assign.
    for (const auto &Assign : AssignToRemove)
      Picked.AssignToRemove.insert(Assign);

    // If we reach this point I is has not been picked for serialization yet, it
    // isn't a statement, and MemoryRead is available to all users of I either
    // directly or through some other local variable where the whole I is
    // available.
    // We have to recur in DFS fashion only on those uses for which we're using
    // MemoryRead directly (not through another local variable where I is
    // available).
    SmallSet<Instruction *, 8> UsersThatRequireMemoryReadSerialized;
    revng_log(Log, "recur on users that aren't available");
    for (Use &TheUse : I->uses()) {

      if (auto It = Picked.ToReplaceWithAvailable.find(&TheUse);
          It != Picked.ToReplaceWithAvailable.end()) {
        revng_log(Log,
                  "TheUse is available: " << dumpToString(TheUse.getUser()));
        continue;
      }

      User *TheUser = TheUse.getUser();

      auto *UserInstruction = cast<Instruction>(TheUser);
      if (rc_recur shouldSerializeReadBeforeOrAtI(UserInstruction, MemoryRead))
        UsersThatRequireMemoryReadSerialized.insert(UserInstruction);
    }

    size_t NumUsersRequiringSerialization = UsersThatRequireMemoryReadSerialized
                                              .size();

    // If no users require MemoryRead to be serialized before them, there's
    // nothing to do, and I doesn't require MemoryRead to be serialized either.
    if (NumUsersRequiringSerialization == 0) {
      revng_log(Log, "No User requires MemoryRead to be serialized");
      rc_return false;
    }

    if (I == MemoryRead) {
      revng_log(Log, "I == MemoryRead: Picked.ToSerialize.insert(I)");
      revng_assert(not IType->isVoidTy() and not IType->isAggregateType()
                   and not isCallToTagged(I, FunctionTags::IsRef));
      Picked.ToSerialize.insert(I);
      rc_return false;
    }

    // If some users of I require MemoryRead to be serialized before them,
    // just serialize I.
    revng_log(Log, "Some of I's users require MemoryRead to be serialized");
    revng_log(Log, "Try and serialize I");
    if (IType->isVoidTy() or IType->isAggregateType()
        or isCallToTagged(I, FunctionTags::IsRef)) {
      revng_log(Log, "I can't be serialized, propagate up.");
      rc_return true;
    } else {
      revng_log(Log, "Picked.ToSerialize.insert(I)");
      Picked.ToSerialize.insert(I);
      rc_return false;
    }

    revng_abort();
    rc_return true;
  }

  void pickFrom(Instruction *I, Instruction *MemoryRead) {
    shouldSerializeReadBeforeOrAtI(I, MemoryRead);
  }

private:
  Function &F;
  const PPGWithInstructionMap &Graph;
  const ResultMap &MFPResultMap;
  PickedInstructions Picked;
  std::unordered_map<const Instruction *, size_t> ProgramOrdering;
};

using TypeMap = std::map<const Value *, const model::UpcastableType>;

template<bool IsLegacy>
using LVB = LocalVariableBuilder<IsLegacy>;

template<bool IsLegacy>
class VariableInserter {
public:
  using PickedInstructions = PickedInstructions<IsLegacy>;

public:
  VariableInserter(Function &TheF,
                   const model::Binary &TheModel,
                   TypeMap &&TMap) :
    Model(TheModel),
    TheTypeMap(std::move(TMap)),
    F(TheF),
    LocalVariableBuilder(LVB<IsLegacy>::make(TheModel, TheF)) {}

public:
  bool run(const PickedInstructions &Picked) {
    LocalVariableBuilder.setTargetFunction(&F);

    bool Changed = false;

    for (const auto &[TheUse, TheAssign] : Picked.ToReplaceWithAvailable) {
      auto *Copy = LocalVariableBuilder.createCopyFromAssignedOnUse(TheAssign,
                                                                    *TheUse);
      TheUse->set(Copy);
    }

    for (Instruction *I : Picked.AssignToRemove) {
      Changed = true;
      I->eraseFromParent();
    }

    for (Instruction *I : Picked.ToSerialize)
      Changed |= serializeToLocalVariable(I);

    return Changed;
  }

private:
  bool serializeToLocalVariable(Instruction *I);

  bool shouldReplaceUseWithCopies(const Instruction *I, const Use &U) const;

  model::UpcastableType getModelType(const Instruction *I) const {
    if constexpr (IsLegacy) {
      return TheTypeMap.at(I);
    } else {
      auto *IType = I->getType();
      revng_assert(IType->isIntOrPtrTy());
      uint64_t ByteSize = 0ULL;
      if (IType->isIntegerTy()) {
        unsigned NumBits = IType->getIntegerBitWidth();
        revng_assert(NumBits);
        revng_assert(NumBits == 1 or (NumBits % 8 == 0));
        ByteSize = (NumBits == 1) ? 1 : (NumBits / 8);
      } else {
        ByteSize = getPointerSize(Model.Architecture());
      }

      return model::PrimitiveType::make(model::PrimitiveKind::Generic,
                                        ByteSize);
    }
  }

private:
  const model::Binary &Model;
  const TypeMap TheTypeMap;
  Function &F;
  LocalVariableBuilder<IsLegacy> LocalVariableBuilder;
};

template<bool IsLegacy>
using VI = VariableInserter<IsLegacy>;

template<bool IsLegacy>
bool VI<IsLegacy>::shouldReplaceUseWithCopies(const Instruction *I,
                                              const Use &U) const {
  auto *Call = getCallToIsolatedFunction(I);
  if (not Call)
    return true;

  const auto *ProtoT = getCallSitePrototype(Model, cast<CallInst>(I));
  abi::FunctionType::Layout Layout = abi::FunctionType::Layout::make(*ProtoT);

  // If the Isolated function doesn't return an aggregate, we have to
  // inject copies from local variables.
  if (Layout.returnMethod() != abi::FunctionType::ReturnMethod::ModelAggregate)
    return true;

  unsigned NumUses = I->getNumUses();

  // SPTAR return aggregates also need copies from local variables,
  // because they are emitted as scalar pointer variables in C.
  if (Layout.hasSPTAR()) {
    revng_assert(0 == NumUses);
    return true;
  }

  if constexpr (IsLegacy) {
    // Non-SPTAR return aggregates, in legacy mode, are special in many ways:
    // 1. they basically imply a LocalVariable;
    // 2. their only expected use is supposed to be in custom opcodes that
    // expect references
    // For these reasons it would be wrong to inject a Copy.
    if (isCallToTagged(U.getUser(), FunctionTags::Assign)) {
      // If it's an assignment, and the operand number is 0, the value of I is
      // being written somewhere (in the location referenced by operand 1).
      // Hence we have to inject a copy.
      return U.getOperandNo() == 0;
    }
    revng_assert(1 == U.getOperandNo());
    revng_assert(isCallToTagged(U.getUser(), FunctionTags::AddressOf)
                 or isCallToTagged(U.getUser(), FunctionTags::ModelGEPRef));
  } else {
    // Non-SPTAR return aggregates expect at most a single use, which is an
    // assignment of their value into a local variable.
    revng_assert(NumUses < 2);
    if (NumUses) {
      const Use &OnlyUse = *I->uses().begin();
      revng_assert(isa<StoreInst>(OnlyUse.getUser()));
      unsigned OpNum = OnlyUse.getOperandNo();
      revng_assert(OpNum != StoreInst::getPointerOperandIndex());
    }
  }
  return false;
}

template<bool IsLegacy>
bool VariableInserter<IsLegacy>::serializeToLocalVariable(Instruction *I) {
  // We can't serialize instructions with reference semantics into local
  // variables because C doesn't have references.
  revng_assert(not isCallToTagged(I, FunctionTags::IsRef));

  // Compute the model type returned from the call.
  revng_assert(I->getType()->isIntOrPtrTy());
  const model::UpcastableType &VariableType = getModelType(I);

  const llvm::DataLayout &DL = I->getModule()->getDataLayout();
  auto ModelSize = VariableType->size().value();
  auto *IType = I->getType();
  auto IRSize = DL.getTypeStoreSize(IType);
  if (ModelSize < IRSize) {
    revng_assert(IType->isPointerTy());
    using model::Architecture::getPointerSize;
    auto PtrSize = getPointerSize(Model.Architecture());
    revng_assert(ModelSize == PtrSize);
  } else if (ModelSize > IRSize) {
    auto &Prototype = *getCallSitePrototype(Model, cast<CallInst>(I));
    using namespace abi::FunctionType;
    abi::FunctionType::Layout Layout = Layout::make(Prototype);
    revng_assert(Layout.returnMethod() == ReturnMethod::ModelAggregate);
    if (Layout.hasSPTAR())
      revng_assert(0 == I->getNumUses());
  }

  // First, we have to declare the LocalVariable, always at the entry block.
  // Create instruction that allocates a LocalVariable
  LocalVarType<IsLegacy> *LocalVariable = LocalVariableBuilder
                                            .createLocalVariable(*VariableType);

  // Then, we have to replace all the uses of I so that they make a Copy
  // from the LocalVariable, unless it's a call to an IsolatedFunction that
  // already returns a local variable, in which case we don't have to do
  // anything with uses.
  for (Use &U : llvm::make_early_inc_range(I->uses())) {
    revng_assert(isa<Instruction>(U.getUser()));

    llvm::Value *ValueToUse = LocalVariable;
    if (shouldReplaceUseWithCopies(I, U)) {
      ValueToUse = LocalVariableBuilder.createCopyOnUse(LocalVariable, U);
    }
    U.set(ValueToUse);
  }

  LocalVariableBuilder.createAssignmentBefore(LocalVariable,
                                              I,
                                              I->getNextNonDebugInstruction());

  return true;
}

template<bool IsLegacy>
struct SwitchToStatements : public FunctionPass {
public:
  using PPGWithInstructionMap = PPGWithInstructionMap<IsLegacy>;
  using ResultMap = ResultMap<IsLegacy>;

public:
  static char ID;

  SwitchToStatements() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LoadModelWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    revng_log(Log, "SwitchToStatements: " << F.getName());

    auto Graph = PPGWithInstructionMap::makeFromFunction(F);

    ResultMap Result = getMFP<IsLegacy>(&Graph.ProgramPointsGraph);

    auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();
    const TupleTree<model::Binary> &Model = ModelWrapper.getReadOnlyModel();

    auto ModelFunction = llvmToModelFunction(*Model, F);
    revng_assert(ModelFunction != nullptr);

    InstructionToSerializePicker InstructionPicker{ F, Graph, Result };

    TypeMap InstructionTypes = {};
    if constexpr (IsLegacy) {
      InstructionTypes = initModelTypesConsideringUses(F,
                                                       ModelFunction,
                                                       *Model,
                                                       /* PointersOnly */
                                                       false);
    }
    VariableInserter<IsLegacy> VarInserter{ F,
                                            *Model,
                                            std::move(InstructionTypes) };

    bool Changed = VarInserter.run(InstructionPicker.pick());

    return Changed;
  }
};

template<>
char SwitchToStatements<false>::ID = 0;

template<>
char SwitchToStatements<true>::ID = 0;

using RegisterLegacy = RegisterPass<SwitchToStatements<true>>;
static RegisterLegacy
  X("legacy-switch-to-statements", "LegacySwitchToStatements", false, false);

using Register = RegisterPass<SwitchToStatements<false>>;
static Register Y("switch-to-statements", "SwitchToStatements", false, false);
