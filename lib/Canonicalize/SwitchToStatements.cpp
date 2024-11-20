//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>
#include <functional>
#include <map>
#include <memory_resource>
#include <set>
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
#include "llvm/Pass.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ABI/ModelHelpers.h"
#include "revng/ADT/GenericGraph.h"
#include "revng/ADT/SmallMap.h"
#include "revng/InitModelTypes/InitModelTypes.h"
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

struct SwitchToStatements : public FunctionPass {
public:
  static char ID;

  SwitchToStatements() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LoadModelWrapperPass>();
  }

  bool runOnFunction(Function &F) override;
};

struct AvailableExpression {
  // The expression that is available
  Instruction *Expression = nullptr;

  // The Assign call that has assigned the Expression to some location.
  // It can be used to retrieve the address of the location itself.
  // nullptr means that we don't have a specific address but the Expression
  // itself can be computed at the given program point without breaking
  // semantics.
  // We need to assign a semantic to nullptr for CallInst and Copy, which are
  // note necessarily assigned to any location by an Assign call.
  CallInst *Assign = nullptr;

  bool operator==(const AvailableExpression &) const = default;
  std::strong_ordering operator<=>(const AvailableExpression &) const = default;
};

using AvailableSet = std::set<AvailableExpression>;

constexpr size_t SmallSize = 8;
using InstructionVector = SmallVector<Instruction *, SmallSize>;
using InstructionSetVector = SmallSetVector<Instruction *, SmallSize>;

struct ProgramPointData {
  Instruction *TheInstruction = nullptr;
  ProgramPointData(Instruction *I) : TheInstruction(I){};
};

static auto findAvailableRange(const AvailableSet &Availables, Instruction *I) {
  auto Begin = Availables.lower_bound(AvailableExpression{ .Expression = I,
                                                           .Assign = nullptr });
  auto End = Availables.upper_bound(AvailableExpression{
    .Expression = std::next(I), .Assign = nullptr });
  return llvm::make_range(Begin, End);
}

using ProgramPointNode = BidirectionalNode<ProgramPointData>;
using ProgramPointsCFG = GenericGraph<ProgramPointNode>;

struct AvailableExpressionsAnalysis;
using ALA = AvailableExpressionsAnalysis;

struct AvailableExpressionsAnalysis {
  using GraphType = ProgramPointsCFG *;
  using LatticeElement = AvailableSet;
  using Label = ProgramPointNode *;
  using MFPResult = MFP::MFPResult<ALA::LatticeElement>;

  ALA::LatticeElement combineValues(const ALA::LatticeElement &LHS,
                                    const ALA::LatticeElement &RHS) const {
    return SetIntersectionLattice<LatticeElement>::combineValues(LHS, RHS);
  }

  bool isLessOrEqual(const ALA::LatticeElement &LHS,
                     const ALA::LatticeElement &RHS) const {
    return SetIntersectionLattice<LatticeElement>::isLessOrEqual(LHS, RHS);
  }

  ALA::LatticeElement applyTransferFunction(ProgramPointNode *L,
                                            const ALA::LatticeElement &E) const;
};

using LatticeElement = ALA::LatticeElement;
using MFPResult = ALA::MFPResult;
using ResultMap = std::map<ProgramPointNode *, MFPResult>;

static bool isStatement(const Instruction *I) {
  // TODO: this is workaround for SelectInst being often involved in nasty
  // huge dataflows.
  // In the future we should drop this from here and add a separate pass after
  // this, that takes care of forcing local variables for nasty dataflows.
  if (isa<SelectInst>(I))
    return true;

  return hasSideEffects(*I);
}

static bool isProgramPoint(const Instruction *I) {

  // TODO: In the future this pass will have to be updated to handle
  // Load/Store/Alloca instead of Copy/Assign/LocalVariable, in order to be
  // able to use LLVM's alias analysis.
  // For now we just assume that we don't have Load/Store/Alloca at all.
  // Whenever we'll do the switchover, we'll have to replace all the logic
  // of Copy/Assign/LocalVariable with Load/Store/Alloca, and just drop
  // everything related to Copy/Assign/LocalVariable.
  // PHINodes will have to be dealt with if/when we move this pass before
  // ExitSSA.
  revng_assert(not isa<LoadInst>(I) and not isa<StoreInst>(I)
               and not isa<AllocaInst>(I) and not isa<PHINode>(I));

  return I == &I->getParent()->front() or isStatement(I) or mayReadMemory(*I);
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

static std::optional<const Value *>
getAccessedLocalVariable(const Instruction *I) {

  // If it's not a Copy not an Assign then it's not an access to a local
  // variable.
  const CallInst *CallToCopy = getCallToTagged(I, FunctionTags::Copy);
  const CallInst *CallToAssign = getCallToTagged(I, FunctionTags::Assign);
  if (not CallToCopy and not CallToAssign)
    return std::nullopt;

  const CallInst *AccessCall = CallToCopy ? CallToCopy : CallToAssign;

  unsigned AccessArgumentNumber = CallToAssign ? 1 : 0;
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
}

static bool localVariablesNoAlias(const Instruction *I, const Instruction *J) {

  // Copies from local variables never alias anyone else, except other
  // instructions that copy or assign the same local variable
  std::optional<const Value *> MayBeAccessedByI = getAccessedLocalVariable(I);
  std::optional<const Value *> MayBeAccessedByJ = getAccessedLocalVariable(J);

  // If either doesn't access a local variable, they are noAlias.
  if (not MayBeAccessedByI.has_value() or not MayBeAccessedByJ.has_value())
    return true;

  const Value *AccessedByI = *MayBeAccessedByI;
  const Value *AccessedByJ = *MayBeAccessedByJ;

  // If either is nullptr, there is at least one among I and J that access many
  // variables, and we just can't say with certainty that they are noAlias
  if (nullptr == AccessedByI or nullptr == AccessedByJ)
    return false;

  // For all the other cases they are noAlias only if the accessed
  // local variable is different.
  return AccessedByI != AccessedByJ;
}

static bool doesNotAccessMemory(const Instruction *I) {
  auto *Call = dyn_cast_or_null<CallInst>(I);
  return Call and Call->getMemoryEffects().doesNotAccessMemory();
}

static bool noAlias(const Instruction *I, const Instruction *J) {
  revng_log(Log, "noAlias?");
  LoggerIndent X{ Log };
  revng_log(Log, "I: " << dumpToString(I));
  revng_log(Log, "J: " << dumpToString(J));
  LoggerIndent XX{ Log };
  // If either instruction doesn't access memory, they are noAlias for sure.
  if (doesNotAccessMemory(I) or doesNotAccessMemory(J)) {
    revng_log(Log, "I or J doesNotAccessMemory");
    return true;
  }

  // Here both instructions access memory.

  // First, handle LocalVariables specifically.
  // TODO: this is a poor's man alias analysis, which only explicitly handles
  // stuff that is frequent and that we care about. In the future we have plans
  // to replace it with a full fledged AliasAnalysis from LLVM
  if (localVariablesNoAlias(I, J)) {
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

static void applyTransferFunction(Instruction *I, LatticeElement &E) {

  revng_log(Log, "applyTransferFunction on Instruction: " << dumpToString(I));
  LoggerIndent X{ Log };

  if (isStatement(I)) {
    revng_log(Log, "isStatement");
    LoggerIndent XX{ Log };
    for (const AvailableExpression &A : llvm::make_early_inc_range(E)) {
      const auto &[Available, Assign] = A;
      revng_log(Log, "Available: " << dumpToString(Available));
      revng_log(Log, "Assign: " << dumpToString(Assign));
      LoggerIndent XXX{ Log };
      if (not noAlias(I, Available)) {
        revng_log(Log, "Available: " << dumpToString(Available));
        revng_log(Log, "is not noAlias (MayAlias) with I");
        revng_log(Log, "erase Available");
        E.erase(A);
      } else if (not noAlias(I, Assign)) {
        revng_log(Log, "Assign: " << dumpToString(Assign));
        revng_log(Log, "erase Available");
        E.erase(A);
      } else {
        revng_log(Log, "is noAlias with I");
      }
    }
  }

  if (auto *Assign = getCallToTagged(I, FunctionTags::Assign)) {
    if (isa<Instruction>(Assign->getArgOperand(0))) {
      revng_log(Log, "I is Assign");
      revng_log(Log,
                "insert Available: " << dumpToString(Assign->getArgOperand(0)));
      revng_log(Log, "       Assign: " << dumpToString(Assign));
      E.insert(AvailableExpression{
        .Expression = cast<Instruction>(Assign->getArgOperand(0)),
        .Assign = Assign,
      });
    }
  }

  if (mayReadMemory(*I)) {
    revng_log(Log, "mayReadMemory -> insert Available: I");
    E.insert(AvailableExpression{
      .Expression = I,
      .Assign = nullptr,
    });
  }
}

LatticeElement ALA::applyTransferFunction(ProgramPointNode *ProgramPoint,
                                          const LatticeElement &E) const {

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
struct ProgramPointsGraphWithInstructionMap {
  ProgramPointsCFG ProgramPointsGraph;
  InstructionProgramPoint ProgramPoint;
  InstructionProgramPoint PreviousProgramPointInBlock;
  InstructionProgramPoint NextProgramPointInBlock;

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

using PPGWithInstructionMap = ProgramPointsGraphWithInstructionMap;

static PPGWithInstructionMap
makeProgramPointsWithInstructionsGraph(Function &F) {

  SmallMap<BasicBlock *, std::pair<ProgramPointNode *, ProgramPointNode *>, 8>
    BlockToBeginEndNode;

  PPGWithInstructionMap Result;

  ProgramPointsCFG &TheCFG = Result.ProgramPointsGraph;
  InstructionProgramPoint &ProgramPoint = Result.ProgramPoint;
  InstructionProgramPoint
    &PreviousProgramPointInBlock = Result.PreviousProgramPointInBlock;
  InstructionProgramPoint &NextProgramPointInBlock = Result
                                                       .NextProgramPointInBlock;

  const auto MakeCFGNode = [&TheCFG, &ProgramPoint](Instruction *I) {
    ProgramPointNode *NewNode = TheCFG.addNode(I);
    ProgramPoint[I] = NewNode;
    return NewNode;
  };

  for (BasicBlock &BB : F) {
    InstructionSetVector ProgramPoints = getProgramPoints(BB);

    // Reserve space for the new ProgramPoints. This is for performance but also
    // for stability of pointers while adding new nodes, which allows to also
    // save pointers to begin and end nodes of each block in a map, to handle
    // addition of inter-block edges.
    // If we don't reserve the pointers returned by addNode aren't stable and
    // the trick for adding inter-block edges doesn't work.
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

static ResultMap getMFP(ProgramPointsCFG *TheGraph) {
  AvailableSet Bottom;
  for (ProgramPointNode *N : llvm::nodes(TheGraph)) {
    Instruction *I = N->TheInstruction;

    if (mayReadMemory(*I)) {
      Bottom.insert(AvailableExpression{
        .Expression = I,
        .Assign = nullptr,
      });
    }

    if (auto *Assign = getCallToTagged(I, FunctionTags::Assign)) {
      if (isa<Instruction>(Assign->getArgOperand(0))) {
        Bottom.insert(AvailableExpression{
          .Expression = cast<Instruction>(Assign->getArgOperand(0)),
          .Assign = Assign,
        });
      }
    }
  }

  AvailableSet Empty{};
  return MFP::getMaximalFixedPoint<ALA>({},
                                        TheGraph,
                                        Bottom,
                                        Empty,
                                        { TheGraph->getEntryNode() });
}

struct PickedInstructions {
  SetVector<Instruction *> ToSerialize = {};
  MapVector<Use *, CallInst *> ToReplaceWithAvailable = {};
  SmallPtrSet<CallInst *, 8> AssignToRemove = {};
};

class InstructionToSerializePicker {
public:
  InstructionToSerializePicker(Function &TheF,
                               const ProgramPointsGraphWithInstructionMap
                                 &TheGraph,
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
        if (isStatement(&I) and not I.getType()->isVoidTy()
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
    if (isStatement(I)) {
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

    MapVector<Use *, CallInst *> ToReplaceWithAvailable;
    SmallPtrSet<CallInst *, 8> AssignToRemove;

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

      auto *UserAssignCall = getCallToTagged(UserInstruction,
                                             FunctionTags::Assign);
      if (UserAssignCall) {
        // Skip over the Assign operand representing variables that are being
        // assigned, because we needwant to preserve them.
        if (UserAssignCall->isArgOperand(&U)
            and UserAssignCall->getArgOperandNo(&U) == 1) {
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

          CallInst *Selected = nullptr;
          size_t ProgramOrder = std::numeric_limits<size_t>::max();
          for (const AvailableExpression &A : AvailableRange) {
            if (nullptr == A.Assign)
              continue;

            size_t NewProgramOrder = ProgramOrdering.at(A.Assign);
            if (NewProgramOrder < ProgramOrder) {
              ProgramOrder = NewProgramOrder;
              Selected = A.Assign;
            }
          }

          revng_log(Log, "Selected: " << dumpToString(Selected));
          if (not Selected) {
            revng_log(Log, "Selected is not an assignment. Serialize I");
            rc_return SerializeI();
          }

          revng_assert(isCallToTagged(Selected, FunctionTags::Assign));

          if (not UserAssignCall) {
            ToReplaceWithAvailable[&U] = Selected;
            continue;
          }

          std::optional<const Value *>
            MayBeAccessedByUser = getAccessedLocalVariable(UserAssignCall);
          std::optional<const Value *>
            MayBeAccessedBySelected = getAccessedLocalVariable(Selected);

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
  const ProgramPointsGraphWithInstructionMap &Graph;
  const ResultMap &MFPResultMap;
  PickedInstructions Picked;
  std::unordered_map<const Instruction *, size_t> ProgramOrdering;
};

using TypeMap = std::map<const Value *, const model::UpcastableType>;

class VariableBuilder {
public:
  VariableBuilder(Function &TheF,
                  const model::Binary &TheModel,
                  TypeMap &&TMap) :
    Model(TheModel),
    TheTypeMap(std::move(TMap)),
    F(TheF),
    Builder(TheF.getContext()),
    LocalVarPool(FunctionTags::LocalVariable.getPool(*TheF.getParent())),
    AssignPool(FunctionTags::Assign.getPool(*TheF.getParent())),
    CopyPool(FunctionTags::Copy.getPool(*TheF.getParent())) {}

public:
  bool run(const PickedInstructions &Picked) {

    bool Changed = false;

    for (const auto &[TheUse, TheAssign] : Picked.ToReplaceWithAvailable) {
      auto *UserInstruction = cast<Instruction>(TheUse->getUser());
      Builder.SetInsertPoint(UserInstruction);

      auto *TheAddress = cast<CallInst>(TheAssign)->getArgOperand(1);

      // Create a Copy to dereference TheAssign
      auto *CopyFnType = getCopyType(TheAddress->getType());
      auto *CopyFunction = CopyPool.get(TheAddress->getType(),
                                        CopyFnType,
                                        "Copy");
      auto *Copy = Builder.CreateCall(CopyFunction, { TheAddress });
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

  bool usesNeedToBeReplacedWithCopiesFromLocal(const Instruction *I) const;

private:
  const model::Binary &Model;
  const TypeMap TheTypeMap;
  Function &F;
  IRBuilder<> Builder;
  OpaqueFunctionsPool<Type *> LocalVarPool;
  OpaqueFunctionsPool<Type *> AssignPool;
  OpaqueFunctionsPool<Type *> CopyPool;
};

bool VariableBuilder::usesNeedToBeReplacedWithCopiesFromLocal(const Instruction
                                                                *I) const {
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

  // Non-SPTAR return aggregates are special in many ways:
  // 1. they basically imply a LocalVariable;
  // 2. their only expected use is supposed to be in custom opcodes that
  // expect
  //    references
  // For these reasons it would be wrong to inject a Copy.
  for (const llvm::Use &U : I->uses()) {
    revng_assert(1 == U.getOperandNo()
                 and (isCallToTagged(U.getUser(), FunctionTags::AddressOf)
                      or isCallToTagged(U.getUser(), FunctionTags::ModelGEPRef)
                      or isCallToTagged(U.getUser(), FunctionTags::Assign)));
  }
  return false;
}

bool VariableBuilder::serializeToLocalVariable(Instruction *I) {
  // We can't serialize instructions with reference semantics into local
  // variables because C doesn't have references.
  revng_assert(not isCallToTagged(I, FunctionTags::IsRef));

  // First, we have to declare the LocalVariable, always at the entry block.
  Builder.SetInsertPoint(&F.getEntryBlock().front());

  auto *IType = I->getType();
  auto *LocalVarFunctionType = getLocalVarType(IType);
  auto *LocalVarFunction = LocalVarPool.get(IType,
                                            LocalVarFunctionType,
                                            "LocalVariable");

  // Compute the model type returned from the call.
  const model::UpcastableType &VariableType = TheTypeMap.at(I);
  const llvm::DataLayout &DL = I->getModule()->getDataLayout();
  auto ModelSize = VariableType->size().value();
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

  // TODO: until we don't properly handle variable declarations with inline
  // initialization (might need MLIR), we cannot declare const local
  // variables, because their initialization (which is forcibly out-of-line)
  // would assign them and fail to compile.
  // For this reason if at this point we're trying to declare a constant
  // local variable, we're forced to throw away the constness information.
  Constant *TString = toLLVMString(model::getNonConst(*VariableType),
                                   *F.getParent());

  // Inject call to LocalVariable
  CallInst *LocalVarCall = Builder.CreateCall(LocalVarFunction, { TString });

  // Then, we have to replace all the uses of I so that they make a Copy
  // from the LocalVariable, unless it's a call to an IsolatedFunction that
  // already returns a local variable, in which case we don't have to do
  // anything with uses.
  bool DoCopy = usesNeedToBeReplacedWithCopiesFromLocal(I);
  for (Use &U : llvm::make_early_inc_range(I->uses())) {
    revng_assert(isa<Instruction>(U.getUser()));
    Builder.SetInsertPoint(cast<Instruction>(U.getUser()));

    llvm::Value *ValueToUse = LocalVarCall;
    if (DoCopy) {
      // Create a Copy to dereference the LocalVariable
      auto *CopyFnType = getCopyType(LocalVarCall->getType());
      auto *CopyFunction = CopyPool.get(LocalVarCall->getType(),
                                        CopyFnType,
                                        "Copy");
      ValueToUse = Builder.CreateCall(CopyFunction, { LocalVarCall });
    }
    U.set(ValueToUse);
  }

  // We have to assign the result of I to the local variable, right
  // after I itself.
  Builder.SetInsertPoint(I->getParent(), std::next(I->getIterator()));

  // Inject Assign() function
  auto *AssignFnType = getAssignFunctionType(IType, LocalVarCall->getType());
  auto *AssignFunction = AssignPool.get(IType, AssignFnType, "Assign");

  Builder.CreateCall(AssignFunction, { I, LocalVarCall });

  return true;
}

bool SwitchToStatements::runOnFunction(Function &F) {

  revng_log(Log, "SwitchToStatements: " << F.getName());

  ProgramPointsGraphWithInstructionMap
    Graph = makeProgramPointsWithInstructionsGraph(F);

  ResultMap Result = getMFP(&Graph.ProgramPointsGraph);

  auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();
  const TupleTree<model::Binary> &Model = ModelWrapper.getReadOnlyModel();

  auto ModelFunction = llvmToModelFunction(*Model, F);
  revng_assert(ModelFunction != nullptr);

  InstructionToSerializePicker InstructionPicker{ F, Graph, Result };
  VariableBuilder VarBuilder{ F,
                              *Model,
                              initModelTypes(F,
                                             ModelFunction,
                                             *Model,
                                             /*PointerOnly*/ false) };

  bool Changed = VarBuilder.run(InstructionPicker.pick());

  return Changed;
}

char SwitchToStatements::ID = 0;

using Register = RegisterPass<SwitchToStatements>;
static Register X("switch-to-statements", "SwitchToStatements", false, false);
