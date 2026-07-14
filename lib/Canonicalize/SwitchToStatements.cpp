//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <memory_resource>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/CodeGen/CodeGenPassBuilder.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/PassInfo.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ModRef.h"
#include "llvm/Support/TypeSize.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ABI/ModelHelpers.h"
#include "revng/ADT/GenericGraph.h"
#include "revng/ADT/SmallMap.h"
#include "revng/Canonicalize/SwitchToStatements.h"
#include "revng/InitModelTypes/InitModelTypes.h"
#include "revng/LocalVariables/LocalVariableBuilder.h"
#include "revng/MFP/MFP.h"
#include "revng/MFP/SetLattices.h"
#include "revng/Model/Binary.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Support/BlockType.h"
#include "revng/Support/Debug.h"
#include "revng/Support/DecompilationHelpers.h"
#include "revng/Support/IRBuilder.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/Tag.h"

static Logger Log{ "switch-to-statements" };

using namespace llvm;

using AssignType = StoreInst;

using CopyType = LoadInst;

using LocalVarType = AllocaInst;

//
// Helper for getting the value operand of an assign (store) instruction.
//

static Use *getStoreValueOperandUse(Instruction *I) {
  if (not I)
    return nullptr;

  if (auto *Assign = dyn_cast<StoreInst>(I))
    return &Assign->getOperandUse(0);
  return nullptr;
}

static Value *getStoreValueOperand(Instruction *I) {
  Use *U = getStoreValueOperandUse(I);
  return U ? U->get() : nullptr;
}

//
// Helpers for statements and side effects.
//

static bool doesNotAccessMemory(const Instruction *I) {
  // We have to hardcode revng_call_stack_arguments and revng_stack_frame
  // because SegregateStackAccesses has to mark them as functions that read
  // inaccessible memory, in order to prevent some LLVM optimizations.
  // Same for OpaqueExtractValue.
  if (auto *Call = dyn_cast<CallInst>(I)) {
    if (Function *Callee = getCalledFunction(Call)) {
      StringRef Name = Callee->getName();
      if (Name.startswith("revng_call_stack_arguments")
          or Name.startswith("revng_stack_frame")) {
        return true;
      }
    }
    if (getCallToTagged(I, FunctionTags::OpaqueExtractValue))
      return true;
    if (getCallToTagged(I, FunctionTags::StructInitializer))
      return true;
  }
  return false;
}

static bool mayHaveSideEffects(const Instruction *I) {
  if (doesNotAccessMemory(I))
    return false;

  return I->mayHaveSideEffects();
}

static bool mayReadMemory(const Instruction *I) {
  if (doesNotAccessMemory(I))
    return false;

  return I->mayReadFromMemory();
}

/// A class that represents an available expression, along with the assignment
/// that writes its value somewhere, making it available.
struct AvailableExpression {
  using AssignType = ::AssignType;

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

using AvailableSet = std::set<AvailableExpression>;

static auto findAvailableRange(const AvailableSet &Availables, Instruction *I) {
  auto Begin = Availables.lower_bound(AvailableExpression{
    .Expression = I, .Assignment = nullptr });
  auto End = Availables.upper_bound(AvailableExpression{
    .Expression = std::next(I), .Assignment = nullptr });
  return llvm::make_range(Begin, End);
}

constexpr size_t SmallSize = 8;
using InstructionVector = SmallVector<Instruction *, SmallSize>;
using InstructionSetVector = SmallSetVector<Instruction *, SmallSize>;

struct ProgramPointData {
  // The program point: the last instruction of the (possibly coalesced) run of
  // instructions it represents.
  Instruction *TheInstruction = nullptr;
  // The first instruction of the run. Equal to TheInstruction for ordinary,
  // non-coalesced program points.
  Instruction *FirstInstruction = nullptr;
  ProgramPointData(Instruction *I) : TheInstruction(I), FirstInstruction(I){};
  ProgramPointData(Instruction *First, Instruction *Last) :
    TheInstruction(Last), FirstInstruction(First){};
};

using ProgramPointNode = BidirectionalNode<ProgramPointData>;
using ProgramPointsCFG = GenericGraph<ProgramPointNode>;

struct AvailableExpressionsMonotoneFramework;

struct AvailableExpressionsMonotoneFramework {
public:
  using GraphType = ProgramPointsCFG *;
  using LatticeElement = AvailableSet;
  using Label = ProgramPointNode *;

private:
  AliasAnalysis *AA;
  ModuleSlotTracker &MST;

public:
  AvailableExpressionsMonotoneFramework(AliasAnalysis *A,
                                        ModuleSlotTracker &TheMST) :
    AA(A), MST(TheMST) {}

public:
  LatticeElement combineValues(const LatticeElement &LHS,
                               const LatticeElement &RHS) const {
    return SetIntersectionLattice<LatticeElement>::combineValues(LHS, RHS);
  }

  bool isLessOrEqual(const LatticeElement &LHS,
                     const LatticeElement &RHS) const {
    return SetIntersectionLattice<LatticeElement>::isLessOrEqual(LHS, RHS);
  }

  LatticeElement applyTransferFunction(ProgramPointNode *L,
                                       const LatticeElement &E,
                                       mfp::NoExtraState &ExtraState) const;

private:
  void applyTransferFunctionImpl(Instruction *I, LatticeElement &E) const;

  bool mayClobber(Instruction *Access, Instruction *Affected) const;
};

using AEMFP = AvailableExpressionsMonotoneFramework;

using LatticeElement = AEMFP::LatticeElement;

using AvailableExpressionsMap = mfp::MFIResultMap<AEMFP>;

bool AEMFP::mayClobber(Instruction *Access, Instruction *Affected) const {

  revng_log(Log, "mayClobber");
  LoggerIndent Indent{ Log };
  revng_log(Log, "Access: " << dumpToString(Access, MST));
  revng_log(Log, "Affected: " << dumpToString(Affected, MST));
  LoggerIndent MoreIndent{ Log };

  bool Result = isModSet(AA->getModRefInfo(Access, Affected));
  revng_log(Log,
            "Access may " << (Result ? std::string() : std::string("not "))
                          << "clobber Affected");
  return Result;
}

void AEMFP::applyTransferFunctionImpl(Instruction *I, LatticeElement &E) const {

  revng_log(Log,
            "applyTransferFunction on Instruction I: " << dumpToString(I, MST));
  LoggerIndent Indent{ Log };

  revng_assert(not isCallToTagged(I, FunctionTags::Copy)
               and not isCallToTagged(I, FunctionTags::Assign));

  if (mayHaveSideEffects(I)) {
    revng_log(Log, "mayHaveSideEffects");
    LoggerIndent XX{ Log };
    for (const AvailableExpression &A : llvm::make_early_inc_range(E)) {
      const auto &[Available, Assign] = A;
      revng_log(Log, "Available: " << dumpToString(Available, MST));
      revng_log(Log, "Assign: " << dumpToString(Assign, MST));
      LoggerIndent XXX{ Log };
      if (mayClobber(I, Available)) {
        revng_log(Log,
                  "I may clobber Available: " << dumpToString(Available, MST));
        revng_log(Log, "erase Available");
        E.erase(A);
      } else if (Assign and mayClobber(I, Assign)) {
        revng_log(Log, "I may clobber Assign: " << dumpToString(Assign, MST));
        revng_log(Log, "erase Available");
        E.erase(A);
      }
    }
  }

  auto *StoredOperand = getStoreValueOperand(I);
  auto *AssignedInstruction = dyn_cast_or_null<Instruction>(StoredOperand);
  if (AssignedInstruction) {
    revng_log(Log, "I is Assign");
    revng_log(Log,
              "insert Available: " << dumpToString(AssignedInstruction, MST));
    revng_log(Log, "       Assign: " << dumpToString(I, MST));

    E.insert(AvailableExpression{
      .Expression = AssignedInstruction,
      .Assignment = cast<AssignType>(I),
    });
  }

  if (mayReadMemory(I)) {
    revng_log(Log, "mayReadMemory -> insert Available: I");
    E.insert(AvailableExpression{
      .Expression = I,
      .Assignment = nullptr,
    });
  }
}

AEMFP::LatticeElement
AEMFP::applyTransferFunction(ProgramPointNode *ProgramPoint,
                             const AEMFP::LatticeElement &E,
                             mfp::NoExtraState &ExtraState) const {

  Instruction *First = ProgramPoint->FirstInstruction;
  Instruction *Last = ProgramPoint->TheInstruction;

  revng_log(Log,
            "applyTransferFunction on ProgramPoint: " << dumpToString(Last,
                                                                      MST));
  LoggerIndent Indent{ Log };

  LatticeElement Result = E;

  revng_log(Log, "initial set");
  if (Log.isEnabled()) {
    LoggerIndent ModeIndent{ Log };
    for (const auto &[Available, Assign] : Result) {
      revng_log(Log, "Available: " << dumpToString(Available, MST));
      revng_log(Log, "Assign: " << dumpToString(Assign, MST));
    }
  }

  // Replay every instruction in the (possibly coalesced) run, from the first to
  // the program point itself, updating Result in place; the set is stored just
  // once per run, which is what the memory optimisation buys. Last is never a
  // terminator, so the instruction past it always exists.
  Instruction *End = Last->getNextNode();
  for (Instruction *I = First; I != End; I = I->getNextNode())
    applyTransferFunctionImpl(I, Result);

  revng_log(Log, "final set");
  if (Log.isEnabled()) {
    LoggerIndent ModeIndent{ Log };
    for (const auto &[Available, Assign] : Result) {
      revng_log(Log, "Available: " << dumpToString(Available, MST));
      revng_log(Log, "Assign: " << dumpToString(Assign, MST));
    }
  }

  return Result;
}

//
// Helpers for identifying program points that are relevant for available
// expressions.
//

// Consecutive memory accesses of the same kind can share a single program
// point - and therefore a single stored set of available expressions -
// instead of getting one each. This is purely a runtime-memory
// optimisation: it does not affect correctness and the emitted code is
// unchanged.
//
// Only these two kinds can be merged, because for both of them the set of
// available expressions computed after the whole run is a valid answer for
// any instruction inside the run (see getAvailableAt):
//   - a load only ever adds an available expression, never removes one, so
//     the set only grows along the run;
//   - a store whose stored value is a constant, a global, or an argument
//     adds no available expression at all (only a store of a value computed
//     by an instruction would; see applyTransferFunctionImpl), so it can
//     only remove expressions and the set only shrinks along the run.
enum class CoalescibleKind {
  None,
  Load,
  ConstStore
};

static CoalescibleKind coalescibleKind(const Instruction *I) {
  if (isa<LoadInst>(I))
    return CoalescibleKind::Load;
  if (const auto *Store = dyn_cast<StoreInst>(I))
    if (not isa<Instruction>(Store->getValueOperand()))
      return CoalescibleKind::ConstStore;
  return CoalescibleKind::None;
}

static bool isProgramPoint(const Instruction *I) {

  const Instruction *UnexpectedInstruction = nullptr;
  // This pass assumes that most custom opcode don't exist. Some of them have
  // been replaced by Load/Store/Alloca, and others have been dropped because in
  // the clift-based pipeline they will be only materialized in Clift as regular
  // operators, so we don't need them in LLVM anymore and we want to make sure
  // they disappear over time until we can actually drop them.
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
      or isCallToTagged(I, FunctionTags::SegmentGlobalGetter)
      or isCallToTagged(I, FunctionTags::UnaryMinus)
      or isCallToTagged(I, FunctionTags::BinaryNot)
      or isCallToTagged(I, FunctionTags::BooleanNot)) {
    UnexpectedInstruction = I;
  }

  if (nullptr != UnexpectedInstruction) {
    I->dump();
    revng_abort("Unexpected Instruction");
  }

  // Coalesce a run of consecutive same-kind loads or const/global/argument
  // stores: only the last instruction of the run is a program point, the
  // earlier ones are absorbed into it.
  if (CoalescibleKind Kind = coalescibleKind(I);
      Kind != CoalescibleKind::None) {
    const Instruction *Next = I->getNextNode();
    return Next == nullptr or coalescibleKind(Next) != Kind;
  }

  return I == &I->getParent()->front() or mayHaveSideEffects(I)
         or mayReadMemory(I);
}

static InstructionSetVector getProgramPoints(BasicBlock &B) {
  InstructionSetVector Results;
  for (Instruction &I : B)
    if (isProgramPoint(&I))
      Results.insert(&I);
  return Results;
}

using InstructionProgramPoint = std::unordered_map<const Instruction *,
                                                   ProgramPointNode *>;

// An extended version of ProgramPointsCFG, that holds a graph of statements
// points, along with a map from each Instruction to its previous statement.
class AvailableExpressionsResult {
public:
  using AvailableExpression = ::AvailableExpression;
  using AvailableSet = ::AvailableSet;
  using AvailableExpressionsMap = ::AvailableExpressionsMap;

public:
  ProgramPointsCFG ProgramPointsGraph;
  AvailableExpressionsMap AvailableExpressions;

private:
  // Map an Instruction to its associated program point in ProgramPointsGraph
  InstructionProgramPoint ProgramPoint;

  // Map an Instruction to its associated previous program point in
  // ProgramPointsGraph.
  InstructionProgramPoint PreviousProgramPointInBlock;

  // Map an Instruction to its associated next program point in
  // ProgramPointsGraph.
  InstructionProgramPoint NextProgramPointInBlock;

  // Maps each instruction absorbed into a coalesced run to the program point
  // node representing that run, so getAvailableAt can answer it from the node's
  // OutValue. This backs the runtime-memory optimisation of giving a whole run
  // a single program point, and thus a single stored set of expressions.
  InstructionProgramPoint RunMember;

  ModuleSlotTracker &MST;

public:
  AvailableExpressionsResult(ModuleSlotTracker &TheMST) : MST(TheMST) {}

  // Factory from llvm::Function
  static AvailableExpressionsResult makeFromFunction(Function &F,
                                                     ModuleSlotTracker &MST) {

    SmallMap<BasicBlock *, std::pair<ProgramPointNode *, ProgramPointNode *>, 8>
      BlockToBeginEndNode;

    AvailableExpressionsResult Result(MST);

    ProgramPointsCFG &TheCFG = Result.ProgramPointsGraph;
    InstructionProgramPoint &ProgramPoint = Result.ProgramPoint;
    InstructionProgramPoint
      &PreviousProgramPointInBlock = Result.PreviousProgramPointInBlock;
    InstructionProgramPoint
      &NextProgramPointInBlock = Result.NextProgramPointInBlock;
    InstructionProgramPoint &RunMember = Result.RunMember;

    const auto MakeCFGNode = [&TheCFG, &ProgramPoint](Instruction *I) {
      ProgramPointNode *NewNode = TheCFG.addNode(I);
      ProgramPoint[I] = NewNode;
      return NewNode;
    };

    for (BasicBlock &BB : F) {
      InstructionSetVector ProgramPoints = getProgramPoints(BB);

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

      // Record the coalesced run each program point represents: walk back over
      // the absorbed same-kind instructions to find the run's first
      // instruction, store it on the node, and map every run member to the node
      // so getAvailableAt answers them from the node's OutValue.
      for (Instruction *Last : ProgramPoints) {
        if (coalescibleKind(Last) == CoalescibleKind::None)
          continue;

        Instruction *First = Last;
        while (Instruction *Previous = First->getPrevNode()) {
          if (coalescibleKind(Previous) != coalescibleKind(Last)
              or ProgramPoints.contains(Previous))
            break;
          First = Previous;
        }

        if (First == Last)
          continue;

        ProgramPointNode *Node = ProgramPoint.at(Last);
        Node->FirstInstruction = First;
        for (Instruction *I = First; I != Last; I = I->getNextNode())
          RunMember[I] = Node;
        RunMember[Last] = Node;
      }

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
  auto getAvailableAt(Instruction *I, const Instruction *Where) const {

    revng_log(Log, "IsAvailableAt");
    revng_log(Log, "Available?: " << dumpToString(I, MST));
    revng_log(Log, "Where: " << dumpToString(Where, MST));

    auto RunMemberIt = RunMember.find(Where);
    if (RunMemberIt != RunMember.end()) {
      revng_log(Log, "is coalesced run member");

      ProgramPointNode *UserProgramPoint = RunMemberIt->second;
      const AvailableSet &Available = AvailableExpressions.at(UserProgramPoint)
                                        .OutValue;
      return findAvailableRange(Available, I);
    }

    auto ProgramPointIt = ProgramPoint.find(Where);
    if (ProgramPointIt != ProgramPoint.end()) {
      revng_log(Log, "is ProgramPoint");

      ProgramPointNode *UserProgramPoint = ProgramPointIt->second;
      const AvailableSet &Available = AvailableExpressions.at(UserProgramPoint)
                                        .InValue;
      return findAvailableRange(Available, I);
    }

    auto PreviousPointIt = PreviousProgramPointInBlock.find(Where);
    if (PreviousPointIt != PreviousProgramPointInBlock.end()) {
      revng_log(Log, "is NOT ProgramPoint");

      ProgramPointNode *UserProgramPoint = PreviousPointIt->second;
      revng_log(Log,
                "Previous ProgramPoint: "
                  << dumpToString(UserProgramPoint->TheInstruction, MST));
      const AvailableSet &Available = AvailableExpressions.at(UserProgramPoint)
                                        .OutValue;
      return findAvailableRange(Available, I);
    }

    auto NextProgramPointIt = NextProgramPointInBlock.find(Where);
    if (NextProgramPointIt != NextProgramPointInBlock.end()) {
      revng_log(Log, "is before first ProgramPoint in BasicBlock");

      ProgramPointNode *UserProgramPoint = NextProgramPointIt->second;
      revng_log(Log,
                "first ProgramPoint in BasicBlock: "
                  << dumpToString(UserProgramPoint->TheInstruction, MST));
      const AvailableSet &Available = AvailableExpressions.at(UserProgramPoint)
                                        .InValue;
      return findAvailableRange(Available, I);
    }

    revng_abort();
  }

  auto getAvailableAt(Instruction *I, const Use &U) const {
    const auto *UserInstruction = cast<Instruction>(U.getUser());
    return getAvailableAt(I, UserInstruction);
  }

  bool isAvailableAt(Instruction *I, const Instruction *Where) const {
    bool Result = not getAvailableAt(I, Where).empty();
    revng_log(Log, "Result: " << Result);
    return Result;
  }

  bool isAvailableAt(Instruction *I, const Use &U) const {
    const auto *UserInstruction = cast<Instruction>(U.getUser());
    return isAvailableAt(I, UserInstruction);
  }
};

using AEResult = AvailableExpressionsResult;

static AEResult getAvailableExpressions(Function &F,
                                        AliasAnalysis *AA,
                                        ModuleSlotTracker &MST) {
  revng_log(Log, "getAvailableExpressions: " << F.getName());

  auto Result = AEResult::makeFromFunction(F, MST);

  // Seed Bottom from every instruction, not from the program point nodes: once
  // runs are coalesced (the memory optimisation) a node no longer maps 1:1 to
  // an instruction, but the lattice domain still needs an entry for every
  // memory read and every instruction-valued store.
  AvailableSet Bottom;
  for (Instruction &Inst : llvm::instructions(F)) {
    Instruction *I = &Inst;

    if (mayReadMemory(I)) {
      Bottom.insert(AvailableExpression{
        .Expression = I,
        .Assignment = nullptr,
      });
    }

    auto *StoredOperand = getStoreValueOperand(I);
    auto *AssignedInstruction = dyn_cast_or_null<Instruction>(StoredOperand);
    if (AssignedInstruction) {
      Bottom.insert(AvailableExpression{
        .Expression = AssignedInstruction,
        .Assignment = cast<AssignType>(I),
      });
    }
  }

  ProgramPointsCFG *Graph = &Result.ProgramPointsGraph;
  ProgramPointNode *Entry = Graph->getEntryNode();

  AEMFP AvailableExpressionsMF{ AA, MST };
  std::vector Entries = { Entry };
  mfp::MFPConfiguration<AEMFP> Configuration{
    .Instance = &AvailableExpressionsMF,
    .Flow = Graph,
    .Bottom = &Bottom,
    .ExtremalLabels = &Entries,
    .EntryLabels = &Entries
  };

  // std::exchange here is only needed to make revng check-conventions happy.
  std::exchange(Result.AvailableExpressions,
                mfp::getMaximalFixedPoint<AEMFP>(Configuration));
  return Result;
}

struct PickedInstructions {
  SetVector<Instruction *> ToSerialize = {};
  MapVector<Use *, AssignType *> ToReplaceWithAvailable = {};
};

// LLVM doesn't ship a function-level analysis that produces a
// ModuleSlotTracker, so we add one here. The Result owns the MST through a
// std::unique_ptr because ModuleSlotTracker is neither copyable nor movable
// (its copy is deleted via a unique_ptr member, and its user-declared virtual
// destructor suppresses the implicit move), and the FAM cache requires the
// Result to be movable.
class ModuleSlotTrackerAnalysis
  : public llvm::AnalysisInfoMixin<ModuleSlotTrackerAnalysis> {

  friend llvm::AnalysisInfoMixin<ModuleSlotTrackerAnalysis>;
  static llvm::AnalysisKey Key;

public:
  struct Result {
    std::unique_ptr<ModuleSlotTracker> MST;
  };

  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &) {
    Result R{ std::make_unique<ModuleSlotTracker>(F.getParent(),
                                                  /* InitMetadata = */ false) };
    R.MST->incorporateFunction(F);
    return R;
  }
};

llvm::AnalysisKey ModuleSlotTrackerAnalysis::Key = {};

class AvailableExpressionsAnalysis
  : public llvm::AnalysisInfoMixin<AvailableExpressionsAnalysis> {

  friend llvm::AnalysisInfoMixin<AvailableExpressionsAnalysis>;
  static llvm::AnalysisKey Key;

public:
  using Result = AvailableExpressionsResult;
  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
    AliasAnalysis *AA = &FAM.getResult<AAManager>(F);
    auto &MST = *FAM.getResult<ModuleSlotTrackerAnalysis>(F).MST;
    return getAvailableExpressions(F, AA, MST);
  }
};

AnalysisKey AvailableExpressionsAnalysis::Key = {};

using AEA = AvailableExpressionsAnalysis;

class InstructionToSerializePicker
  : public AnalysisInfoMixin<InstructionToSerializePicker> {
  friend llvm::AnalysisInfoMixin<InstructionToSerializePicker>;
  static llvm::AnalysisKey Key;

public:
  using Result = PickedInstructions;
  using AvailableExpression = ::AvailableExpression;
  using AssignType = ::AssignType;

private:
  const AEResult *AvailableExpressions = nullptr;
  std::unordered_map<const Instruction *, size_t> ProgramOrdering = {};
  Result Picked;
  AliasAnalysis *AA;
  // A pointer (rather than a reference) so the picker can be default
  // constructed by the FAM factory; the MST is fetched and assigned in run().
  ModuleSlotTracker *MST = nullptr;

public:
  InstructionToSerializePicker() :
    AvailableExpressions(nullptr), ProgramOrdering() {}

public:
  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
    AA = &FAM.getResult<AAManager>(F);
    MST = FAM.getResult<ModuleSlotTrackerAnalysis>(F).MST.get();
    AvailableExpressions = &FAM.getResult<AEA>(F);

    Picked = {};
    ProgramOrdering = {};

    return pick(F);
  }

private:
  bool isSerializable(const Instruction &I) const {
    const Type *T = I.getType();
    return not T->isVoidTy();
  }

  void pick(Instruction *I) {
    LoggerIndent Indent{ Log };
    revng_log(Log, "pick(I), I: " << dumpToString(I, *MST));
    Picked.ToSerialize.insert(I);
  };

  bool isPicked(Instruction *I) const { return Picked.ToSerialize.contains(I); }

  Result pick(Function &F) {
    revng_log(Log, "pick: " << F.getName().str());
    LoggerIndent Indent{ Log };

    // Visit in RPO for determinism
    const auto RPO = llvm::ReversePostOrderTraversal(&F);
    size_t NextOrder = 0;
    for (BasicBlock *BB : RPO)
      for (Instruction &I : *BB)
        ProgramOrdering[&I] = NextOrder++;

    // Start from memory reads, and traverse the dataflow to pick other
    // instructions that need to be serialized.
    for (BasicBlock *BB : RPO)
      for (Instruction &I : *BB)
        if (mayReadMemory(&I))
          pickInstructionForMemoryRead(&I, &I);

    return Picked;
  }

  RecursiveCoroutine<void>
  pickInstructionForMemoryRead(Instruction *I, Instruction *MemoryRead) {
    revng_log(Log, "PickFrom I: " << dumpToString(I, *MST));
    revng_log(Log, "MemoryRead: " << dumpToString(MemoryRead, *MST));

    LoggerIndent Indent{ Log };

    // If I has already been picked we'de done.
    if (isPicked(I)) {
      revng_log(Log, "I isPicked");
      rc_return;
    }

    // If MemoryRead may have side effects, it means that it's a call that
    // accesses memory in read+write fashion. If that happens, in principle we
    // don't have a strong case for picking I. However, if I has more than one
    // use, and we don't pick I, the MemoryRead (which is a call) will end up
    // duplicated in the full expression for I, which is not guaranteed to
    // break semantic. So, whenever MemoryRead may have side effects and I has
    // more than a single use, we have to pick I for preserving semantic.
    if (mayHaveSideEffects(MemoryRead) and I->getNumUses() > 1) {
      revng_log(Log, "I may have side effects, and has many uses");
      pick(I);
      rc_return;
    }

    revng_log(Log, "Check users");
    LoggerIndent UserIndent{ Log };

    MapVector<Use *, AssignType *> ToReplaceWithAvailable;
    SmallVector<Use *> UsesToRecurOn;

    // If we're here, I hasn't been picked for serialization yet.
    // If possible we want to avoid picking it, because our broader goal is to
    // serialize as few instructions as possible.
    //
    // We can avoid serializing I if one of the following applies.
    //
    // 1.
    // MemoryRead is available directly in all uses of I. In this case I can be
    // emitted inline in each of its uses, and MemoryRead will be inlined along
    // with I in every use of I, without breaking semantics.
    // We can avoid serializing I, and recur.
    //
    // 2.
    // For each use U of I where MemoryRead is not available directly,
    // MemoryRead is available indirectly via some assignment S that stored it
    // somewhere.
    // In this case, for each of such use U, we want to make sure that if we
    // inline I in U we will not use MemoryRead directly, but replace the
    // transitive use of MemoryRead in I with a load from the same location as
    // S.
    // However, in practice we have a guarantee that this holds, thanks to
    // recursion. Indeed, if we're reached this place via recursion we must have
    // already processed the transitive use of MemoryRead in I, and we have
    // solved that part of the problem earlier. This means can be sure that
    // whenever we emit I, if MemoryRead is not available directly at I but only
    // via an assignment S, we have already set up things so that we will use a
    // load from S instead of MemoryRead directly to emit I.
    // So we can avoid serializing I, and recur.
    //
    // 3. If MemoryRead is not available at U we check if I itself is available
    // at U via some assignment S that assigned it to some location, where the
    // value of I is still available. If I it's not available via any assignment
    // we have to bail out and pick I for serialization.
    //
    // 4. For each use U of I where MemoryRead is not available, I happens to be
    // available via some assignment S that assigned it to some location, where
    // the value of I is still available.
    //
    // If we're looking at how I is used in U, it means that we've already
    // processed how MemoryRead is used in I, hence by construction the
    // expression that will be emitted in S to compute the value of I will
    // already use the correct value of MemoryRead, either directly or from
    // another assignment T that assigns before I and makes it available
    // indirectly at I.
    //
    // As a result, we just mark the use U to be replaced with the value of I
    // available at S.

    // For each U Use of I where MemoryRead is not available, check if the
    // whole expression represented by I is available at U. If so add it to
    // the ToReplaceWithAvailable.
    // Otherwise if we find even a single use of I where MemoryRead is not
    // available and such that I itself is not available, we have to require I
    // to be either be available somewhere else or be serialized in a new local
    // variable.
    for (Use &U : I->uses()) {
      auto *User = cast<Instruction>(U.getUser());
      revng_log(Log,
                "UseNo: " << U.getOperandNo()
                          << " User: " << dumpToString(User, *MST));
      LoggerIndent MoreUserIndent{ Log };

      // U of I could have already been picked in a previous iteration, from a
      // different memory read.
      // Detect that case and quickly bail out, because in practice we have
      // already taken that decision and we will reuse that.
      // No need to recur here.
      if (Picked.ToReplaceWithAvailable.count(&U)) {
        revng_log(Log,
                  "I was already picked and it's available at User, reading "
                  "from: "
                    << dumpToString(Picked.ToReplaceWithAvailable.lookup(&U),
                                    *MST));
        continue;
      }

      revng_log(Log, "Find where I is available");

      // Case 1. and 2. of the description above.
      // If the MemoryRead is available in U either directly or via an
      // assignment S we're fine and we start looking at the next use.
      if (AvailableExpressions->isAvailableAt(MemoryRead, U)) {
        revng_log(Log, "isAvailableAt(MemoryRead, User)");
        UsesToRecurOn.push_back(&U);
        continue;
      }
      revng_log(Log, "not isAvailableAt(MemoryRead, User)");

      // Case 3. If selectAssignment fails I is not available at U via any
      // assignment, so we pick I and bail out..
      AssignType *SelectedAssign = selectAssignment(I, U);
      if (not SelectedAssign) {
        revng_log(Log, "SelectedAssign: nullptr");
        revng_log(Log, "I is not available at User via other assignments");
        rc_return pick(I);
      }
      revng_log(Log, "SelectedAssign: " << dumpToString(SelectedAssign, *MST));

      // Case 4.
      // If the user is not writing to memory, it cannot interact in any way
      // with the memory written to by SelectedAssign.
      // Just mark U to be replaced from a read from the location assigned by
      // SelectedAssign.
      ToReplaceWithAvailable[&U] = SelectedAssign;
      revng_log(Log, "not mayWriteToMemory(User)");
      // We don't need to recur on uses of U, because all of them will
      // effectively be replaced by reads from SelectedAssign, so the
      // MemoryRead will not be affecting them anymore.
    }

    // If we reach this point, it means that no user forced us to serialize I.
    // At this point we can commit ToReplaceWithAvailable into
    // Picked.ToReplaceWithAvailable.
    for (const auto &Element : ToReplaceWithAvailable)
      Picked.ToReplaceWithAvailable.insert(Element);

    // If we reach this point I is has not been picked for serialization, and
    // MemoryRead is available to all users of I, either directly of via some
    // other local variable where the whole I is available.
    // Hence, we can safely decide that I will not be picked.
    // We still have to check all users of I that are using MemoryRead directly
    // (transitively via I). Those may still be picked.
    revng_log(Log, "Recur on users of I");
    for (Use *U : UsesToRecurOn) {

      auto *User = cast<Instruction>(U->getUser());
      LoggerIndent UserIndent{ Log };
      revng_log(Log,
                "UseNo: " << U->getOperandNo()
                          << " User: " << dumpToString(User, *MST));
      LoggerIndent MoreUserIndent{ Log };
      rc_recur pickInstructionForMemoryRead(User, MemoryRead);
    }
    rc_return;
  }

  /// Returns an assignment where I is available at U
  AssignType *selectAssignment(Instruction *I, Use &U) {
    // If the MemoryRead is not available at U, it may still be the case that
    // I itself is available at U, because it was stored into a pre-existing
    // alloca.
    auto Available = AvailableExpressions->getAvailableAt(I, U);
    if (Available.empty()) {
      revng_log(Log, "I is not available at User");
      return nullptr;
    }
    revng_log(Log, "I is available at User");
    if (Log.isEnabled()) {
      LoggerIndent Indent{ Log };
      for (const AvailableExpression &AE : Available) {
        revng_log(Log, "AE.Available = " << dumpToString(AE.Expression, *MST));
        revng_log(Log, "AE.Assignment = " << dumpToString(AE.Assignment, *MST));
      }
    }

    // Here we have a bunch of places where I is available in U.
    // We want to pick one, so that we will end up replacing the use of I in U
    // with a load from there.

    SmallVector<AssignType *> AssignsWhereIIsAvailable;
    llvm::copy(Available
                 | std::views::transform([](const AvailableExpression &A) {
                     return A.Assignment;
                   })
                 | std::views::filter([](const AssignType *A) {
                     return A != nullptr;
                   }),
               std::back_inserter(AssignsWhereIIsAvailable));

    if (AssignsWhereIIsAvailable.empty())
      return nullptr;

    // Sort them in program order.
    // This is not strictly necessary, but it ensures determinism in picking the
    // candidate.
    // TODO: is this better or worse as an heuristic as opposed to reverse
    // program order, or possibly other heuristics?
    llvm::sort(AssignsWhereIIsAvailable,
               [&PO = ProgramOrdering](const AssignType *LHS,
                                       const AssignType *RHS) {
                 return PO.at(LHS) < PO.at(RHS);
               });

    AssignType *CandidateAssign = AssignsWhereIIsAvailable.front();
    revng_log(Log,
              "First CandidateAssign: " << dumpToString(CandidateAssign, *MST));
    return CandidateAssign;
  }
};

AnalysisKey InstructionToSerializePicker::Key = {};

using LVB = LocalVariableBuilder<false>;

static LocalVariableBuilder<false>
makeVariableBuilder(Function &F, unsigned InputPointerByteSize) {
  VariableBuilderTypes Types = VariableBuilderTypes{ *F.getParent(),
                                                     InputPointerByteSize };

  return LVB::make(Types, &F);
}

class VariableInserter {
public:
  using PickedInstructions = ::PickedInstructions;

private:
  Function &F;

  LocalVariableBuilder<false> VariableBuilder;

public:
  VariableInserter(Function &TheF, unsigned InputPointerByteSize) :
    F(TheF), VariableBuilder(makeVariableBuilder(F, InputPointerByteSize)) {}

public:
  bool run(const PickedInstructions &Picked) {
    bool Changed = false;

    for (const auto &[TheUse, TheAssign] : Picked.ToReplaceWithAvailable)
      TheUse->set(createCopyFromAssignedOnUse(TheAssign, *TheUse));

    for (Instruction *I : Picked.ToSerialize) {
      // If ToSerialize contains something with 0 uses we don't add the
      // local variable. The Clifter will identify this a statement because it
      // has zero uses, and it will turn it into an ExpressionStatement in
      // place, preserving the ordering as if the local variable was emitted.
      if (I->getNumUses() > 0)
        Changed |= serializeToLocalVariable(I);
    }

    return Changed;
  }

private:
  bool serializeToLocalVariable(Instruction *I);

  LocalVarType *createLocalVariableFor(Instruction *I) {
    DebugLoc DL = I->getDebugLoc();
    revng::IRBuilder B(F.getContext());
    B.SetInsertPointPastAllocas(&F, DL);
    return B.createSimpleAlloca(I->getType());
  }

  CopyType *createCopyOnUse(Value *ToCopy, Use &U) {
    // Create a copy from the assigned location at the proper insertion point.
    auto *InsertBefore = cast<Instruction>(U.getUser());
    DebugLoc DL = InsertBefore->getDebugLoc();
    if (auto *I = dyn_cast<Instruction>(ToCopy))
      DL = I->getDebugLoc();
    revng::IRBuilder B(InsertBefore, DL);
    if (auto *Alloca = dyn_cast<AllocaInst>(ToCopy))
      return B.createLoadFromVariable(Alloca, U->getType());
    return B.CreateLoad(U->getType(), ToCopy);
  }

  CopyType *createCopyFromAssignedOnUse(AssignType *Store, Use &U) {
    return createCopyOnUse(Store->getPointerOperand(), U);
  }

  AssignType *createAssignment(LocalVarType *LocalVariable,
                               Instruction *ValueToAssign) {
    auto NextInstruction = ValueToAssign->getNextNonDebugInstruction();
    revng::IRBuilder B(NextInstruction, ValueToAssign->getDebugLoc());
    return B.createStoreToVariable(ValueToAssign, LocalVariable);
  }
};

using VI = VariableInserter;

bool VI::serializeToLocalVariable(Instruction *I) {
  // We can't serialize instructions with reference semantics into local
  // variables because C doesn't have references.
  revng_assert(not isCallToTagged(I, FunctionTags::IsRef));

  // First, we have to declare the LocalVariable, always at the entry block.
  // Create instruction that allocates a LocalVariable
  LocalVarType *LocalVariable = createLocalVariableFor(I);

  // Then, we have to replace all the uses of I so that they make a Copy
  // from the LocalVariable, unless it's a call to an IsolatedFunction that
  // already returns a local variable, in which case we don't have to do
  // anything with uses.
  for (Use &U : llvm::make_early_inc_range(I->uses()))
    U.set(createCopyOnUse(LocalVariable, U));

  createAssignment(LocalVariable, I);
  return true;
}

static void registerAliasAnalysis(FunctionAnalysisManager &FAM) {
  PassBuilder PB;
  PB.registerFunctionAnalyses(FAM);
  FAM.registerPass([] { return llvm::registerAAAnalyses(); });
}

static void registerCommonAnalyses(FunctionAnalysisManager &FAM) {
  FAM.registerPass([] { return ModuleSlotTrackerAnalysis(); });
  FAM.registerPass([] { return AvailableExpressionsAnalysis(); });
  FAM.registerPass([] { return InstructionToSerializePicker(); });
}

class SwitchToStatements : public llvm::PassInfoMixin<SwitchToStatements> {

private:
  /// The size in bytes of a pointer in the Binary we're decompiling.
  /// Necessary for initializing a LocalVariableBuilder.
  unsigned InputPointerByteSize;

public:
  SwitchToStatements(unsigned InputPointerByteSize) :
    InputPointerByteSize(InputPointerByteSize) {}

public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM) {

    VariableInserter VarInserter{ F, InputPointerByteSize };

    const auto &Picked = FAM.getResult<InstructionToSerializePicker>(F);
    bool Changed = VarInserter.run(Picked);

    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }

  static void registerCallbacks(PassBuilder &PB) {
    using PipelineElementArray = ArrayRef<PassBuilder::PipelineElement>;
    PB.registerAnalysisRegistrationCallback(registerAliasAnalysis);
    PB.registerAnalysisRegistrationCallback(registerCommonAnalyses);
    PB.registerPipelineParsingCallback([](StringRef Name,
                                          FunctionPassManager &FPM,
                                          PipelineElementArray) {
      if (Name == "switch-to-statements-test") {
        FPM.addPass(SwitchToStatements{ /*InputPointerSize*/ 8 });
        return true;
      }
      return false;
    });
  }
};

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return { LLVM_PLUGIN_API_VERSION,
           "SwitchToStatementsTests",
           "1.0",
           SwitchToStatements::registerCallbacks };
}

static bool switchToStatements(const model::Binary *Model, llvm::Function &F) {

  revng_log(Log, "switchToStatements: " << F.getName());

  ModuleAnalysisManager MAM;
  FunctionAnalysisManager FAM;

  // Cross register passes, because function-level alias analyses fall back to
  // querying GlobalsAA in some cases. If the analysis manager aren't
  // cross-registered, that fallback query just crashes.
  // We don't activate GlobalsAA, but GlobalsAA still needs to be registered, in
  // order for local AA to query it and see it doesn't have cached results
  // because it did not run.
  FAM.registerPass([&MAM] { return ModuleAnalysisManagerFunctionProxy(MAM); });
  MAM.registerPass([&FAM] { return FunctionAnalysisManagerModuleProxy(FAM); });

  // Register module-level analyses, because alias analysis looks up GlobalsAA,
  // which is a module-level pass.
  PassBuilder PB;
  PB.registerModuleAnalyses(MAM);
  // Register the standard LLVM function analyses (PassInstrumentationAnalysis,
  // ...) and the alias analyses we use further down.
  registerAliasAnalysis(FAM);
  registerCommonAnalyses(FAM);
  FunctionPassManager FPM;
  FPM.addPass(SwitchToStatements(getPointerSize(Model->Architecture())));
  llvm::PreservedAnalyses Preserved = FPM.run(F, FAM);

  return Preserved.areAllPreserved() ? false : true;
}

class SwitchToStatementsPass : public FunctionPass {
public:
  static char ID;

  SwitchToStatementsPass() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LoadModelWrapperPass>();
  }

  bool runOnFunction(Function &F) override;
};

char SwitchToStatementsPass::ID = 0;

bool SwitchToStatementsPass::runOnFunction(llvm::Function &F) {
  auto
    *Model = getAnalysis<LoadModelWrapperPass>().get().getReadOnlyModel().get();
  return switchToStatements(Model, F);
}

using Register = RegisterPass<SwitchToStatementsPass>;
static Register
  Y("switch-to-statements", "SwitchToStatementsPass", false, false);

namespace revng::pypeline::piperuns {

// TODO: inline switchToStatements once we dismiss the old pipeline
void SwitchToStatements::runOnLLVMFunction(const model::Function &Function,
                                           llvm::Function &LLVMFunction) {
  switchToStatements(Model.get(), LLVMFunction);
}

} // namespace revng::pypeline::piperuns
