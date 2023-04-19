//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

/// Analysis that detects which Instructions in a Function need an assignment in
/// decompilation to C.

#include <map>

#include "llvm/IR/Instructions.h"

#include "revng/ADT/ZipMapIterator.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MonotoneFramework.h"

#include "revng-c/Support/DecompilationHelpers.h"
#include "revng-c/Support/FunctionTags.h"

#include "LivenessAnalysis.h"
#include "MarkAssignments.h"

namespace llvm {

class BasicBlock;
class Function;
class Instruction;

} // end namespace llvm

Logger<> MarkLog{ "mark-assignments" };

namespace MarkAssignments {

using TaintSetT = std::set<const llvm::Instruction *>;

static bool haveInterferingSideEffects(const llvm::Instruction *SideEffectful,
                                       const llvm::Instruction &Other,
                                       const TaintSetT &TaintSet) {
  // Branch instructions never have side effects, so no Other could possibly
  // interfere with them.
  if (isa<llvm::BranchInst>(SideEffectful)
      or isa<llvm::SwitchInst>(SideEffectful))
    return false;

  const auto MightInterfere = [SideEffectful](const llvm::Instruction *I) {
    // AddressOf never has side effects.
    if (auto *CallToAddressOf = getCallToTagged(I, FunctionTags::AddressOf)) {
      return false;
    }

    // Copies from local variables never alias anyone else, except other
    // instructions that copy or assign the same local variable
    llvm::CallInst *LocalVar = nullptr;
    bool IsWrite = false;
    if (auto *CallToCopy = getCallToTagged(I, FunctionTags::Copy)) {
      LocalVar = getCallToTagged(CallToCopy->getArgOperand(0),
                                 FunctionTags::LocalVariable);
    } else if (auto *CallToAssign = getCallToTagged(I, FunctionTags::Assign)) {
      LocalVar = getCallToTagged(CallToAssign->getArgOperand(1),
                                 FunctionTags::LocalVariable);
      IsWrite = true;
    }

    llvm::CallInst *OtherLocalVar = nullptr;
    if (auto *OtherCallToCopy = getCallToTagged(SideEffectful,
                                                FunctionTags::Copy)) {
      OtherLocalVar = getCallToTagged(OtherCallToCopy->getArgOperand(0),
                                      FunctionTags::LocalVariable);
    } else if (auto
                 *OtherCallToAssign = getCallToTagged(SideEffectful,
                                                      FunctionTags::Assign)) {
      OtherLocalVar = getCallToTagged(OtherCallToAssign->getArgOperand(1),
                                      FunctionTags::LocalVariable);
      IsWrite = true;
    }

    // If either of the instruction is an access to a local variable, we know
    // that only other accesses to the same local variable can have interfering
    // side effects
    if (LocalVar or OtherLocalVar) {
      // If only one accesses a local variable, then the other does not have
      // interfering side effect for sure.
      if (not LocalVar or not OtherLocalVar)
        return false;
      // If both access the same local variable and at least one is writing,
      // they have interfering side effects
      return IsWrite and (LocalVar == OtherLocalVar);
    }

    if (hasSideEffects(*I))
      return true;

    // TODO: we could check for aliasing between SideEffectful and I
    // here, but it's costly and complicated. We should do that only if
    // necessary.
    if (isa<llvm::LoadInst>(I))
      return true;

    // TODO: we could check for aliasing between SideEffectful and I
    // here, but it's costly and complicated. We should do that only if
    // necessary.
    if (isCallToTagged(I, FunctionTags::ReadsMemory))
      return true;

    return false;
  };

  return MightInterfere(&Other) or llvm::any_of(TaintSet, MightInterfere);
}

class MonotoneTaintMap {
public:
  using Instruction = llvm::Instruction;
  using TaintMap = std::map<Instruction *, std::set<const Instruction *>>;
  using const_iterator = typename TaintMap::const_iterator;
  using iterator = typename TaintMap::iterator;
  using size_type = typename TaintMap::size_type;
  using node_type = typename TaintMap::node_type;

protected:
  TaintMap TaintedPending;
  bool IsBottom;

protected:
  MonotoneTaintMap(const MonotoneTaintMap &) = default;

public:
  MonotoneTaintMap() : TaintedPending(), IsBottom(true){};

  MonotoneTaintMap copy() const { return *this; }
  MonotoneTaintMap &operator=(const MonotoneTaintMap &) = default;

  MonotoneTaintMap(MonotoneTaintMap &&) = default;
  MonotoneTaintMap &operator=(MonotoneTaintMap &&) = default;

public:
  const_iterator begin() const { return TaintedPending.begin(); }
  const_iterator end() const { return TaintedPending.end(); }

  void dump() const { dump(dbg); }

  template<typename O>
  void dump(O &Output) const {
    Output << "{ ";
    for (const auto &[Instr, _] : TaintedPending)
      Output << Instr << " ";
    Output << " }";
  }

public:
  size_type size() const {
    revng_assert(not IsBottom);
    return TaintedPending.size();
  }

  void insertWithTaint(Instruction *Key, TaintSetT &&Taint) {
    revng_assert(not IsBottom);
    auto &TaintSet = TaintedPending[Key];
    TaintSet.insert(Key);
    TaintSet.merge(std::move(Taint));
  }

  const_iterator erase(const_iterator It) {
    revng_assert(not IsBottom);
    return this->TaintedPending.erase(It);
  }

  node_type extract(Instruction *I) { return TaintedPending.extract(I); }

  bool isPending(Instruction *Key) const {
    revng_assert(not IsBottom);
    return TaintedPending.count(Key);
  }

public:
  static MonotoneTaintMap bottom() { return MonotoneTaintMap(); }

  static MonotoneTaintMap top() {
    MonotoneTaintMap Res = {};
    Res.IsBottom = false;
    return Res;
  }

public:
  void combine(const MonotoneTaintMap &Other) {
    if (Other.IsBottom)
      return;

    if (IsBottom) {
      this->TaintedPending = Other.TaintedPending;
      IsBottom = false;
      return;
    }

    std::vector<iterator> ToDrop;

    auto OtherCopy = Other.copy();
    const_iterator OtherEnd = OtherCopy.end();

    // For each element in TaintedPending, check if Other also has it.
    iterator SetIt = this->TaintedPending.begin();
    iterator SetEnd = this->TaintedPending.end();
    while (SetIt != SetEnd) {
      iterator OtherIt = OtherCopy.TaintedPending.find(SetIt->first);
      if (OtherIt == OtherEnd) {
        // If Other does not have the same entry, we drop it.
        ToDrop.push_back(SetIt);
        SetIt = this->TaintedPending.erase(SetIt);
      } else {
        // If Other also has the same entry, we merge the taint sets
        SetIt->second.merge(std::move(OtherIt->second));
        ++SetIt;
      }
    }
  }

  bool lowerThanOrEqual(const MonotoneTaintMap &Other) const {
    if (IsBottom)
      return true;

    if (Other.IsBottom)
      return false;

    if (size() < Other.size())
      return false;

    const auto &Zip = zipmap_range(TaintedPending, Other.TaintedPending);
    for (const auto &PtrPair : Zip) {

      if (nullptr == PtrPair.first)
        return false;

      if (nullptr == PtrPair.second)
        continue;

      revng_assert(PtrPair.first->first == PtrPair.second->first);
      const auto &ThisTaintSet = PtrPair.first->second;
      const auto &OtherTaintSet = PtrPair.second->second;

      if (not std::includes(OtherTaintSet.begin(),
                            OtherTaintSet.end(),
                            ThisTaintSet.begin(),
                            ThisTaintSet.end())) {
        return false;
      }
    }
    return true;
  }
};

using SuccVector = llvm::SmallVector<llvm::BasicBlock *, 2>;

using LatticeElement = MonotoneTaintMap;

class Analysis : public MonotoneFramework<Analysis,
                                          llvm::BasicBlock *,
                                          LatticeElement,
                                          VisitType::ReversePostOrder,
                                          SuccVector> {
private:
  llvm::Function &F;
  AssignmentMap Assignments;
  LivenessAnalysis::LivenessMap LiveIn;

public:
  using Base = MonotoneFramework<Analysis,
                                 llvm::BasicBlock *,
                                 LatticeElement,
                                 VisitType::ReversePostOrder,
                                 SuccVector>;

  using InterruptType = typename Base::InterruptType;

public:
  Analysis(llvm::Function &F) :
    Base(&F.getEntryBlock()), F(F), Assignments(), LiveIn() {
    Base::registerExtremal(&F.getEntryBlock());
  }

  void initialize() {
    Base::initialize();
    LiveIn = computeLiveness(F);
  }

  AssignmentMap &&takeAssignments() { return std::move(Assignments); }

public:
  void assertLowerThanOrEqual(const LatticeElement &A,
                              const LatticeElement &B) const {
    revng_assert(A.lowerThanOrEqual(B));
  }

  [[noreturn]] void dumpFinalState() const { revng_abort(); }

  SuccVector successors(llvm::BasicBlock *BB, InterruptType &) const {
    SuccVector Result;
    for (llvm::BasicBlock *Successor : make_range(succ_begin(BB), succ_end(BB)))
      Result.push_back(Successor);
    return Result;
  }

  std::optional<LatticeElement>
  handleEdge(const LatticeElement &Original,
             const llvm::BasicBlock * /*Source*/,
             const llvm::BasicBlock *Destination) const {

    auto LiveInIt = LiveIn.find(Destination);
    if (LiveInIt == LiveIn.end())
      return std::nullopt;

    const auto &LiveInSet = LiveInIt->second;

    LatticeElement Result = Original.copy();

    const auto IsDead = [&LiveInSet](const llvm::Instruction *I) {
      return not LiveInSet.contains(I);
    };

    for (auto ResultsIt = Result.begin(); ResultsIt != Result.end();) {
      if (IsDead(ResultsIt->first)) {
        ResultsIt = Result.erase(ResultsIt);
      } else {
        ++ResultsIt;
      }
    }

    return Result;
  }

  size_t successor_size(const llvm::BasicBlock *BB, InterruptType &) const {
    return succ_size(BB);
  }

  static LatticeElement extremalValue(const llvm::BasicBlock *) {
    return LatticeElement::top();
  }

  InterruptType transfer(llvm::BasicBlock *BB) {
    using namespace llvm;
    revng_log(MarkLog,
              "transfer: BB in Function: " << BB->getParent()->getName() << '\n'
                                           << BB);

    LatticeElement Pending = this->State[BB].copy();

    for (Instruction &I : *BB) {
      LoggerIndent Indent(MarkLog);
      revng_log(MarkLog,
                "Analyzing Instr: '" << &I << "': " << dumpToString(&I));

      TaintSetT OperandTaintSet;
      {
        // Look at the operands of I.
        // If some of them is still pending, we want to remove them from
        // pending, because at the end of this function we will either mark
        // I as assigned, or insert I in pending.

        revng_log(MarkLog, "Remove operands from pending.");
        LoggerIndent MoreIndent(MarkLog);

        {
          revng_log(MarkLog, "Operands:");
          LoggerIndent EvenMoreMoreIndent(MarkLog);

          for (auto &TheUse : I.operands()) {
            Value *V = TheUse.get();

            revng_log(MarkLog, "Op: '" << V << "': " << dumpToString(V));
            LoggerIndent _(MarkLog);

            if (auto *UsedInstr = dyn_cast<Instruction>(V)) {
              revng_log(MarkLog, "Op is Instruction: erase it from pending");
              auto Handle = Pending.extract(UsedInstr);
              if (not Handle.empty())
                OperandTaintSet.merge(std::move(Handle.mapped()));

            } else {
              revng_log(MarkLog, "Op is NOT Instruction: never in pending");
              revng_assert(isa<Argument>(V) or isa<Constant>(V)
                           or isa<BasicBlock>(V) or isa<MetadataAsValue>(V));
            }
          }
        }
      }

      // After the new redesign of IRCanonicalization PHINodes shouldn't
      // even reach this stage.
      revng_assert(not isa<PHINode>(I));

      // Instructions only allocating a local variable and integer print
      // decorators never need an assignment.
      if (isCallToTagged(&I, FunctionTags::AllocatesLocalVariable)
          || isCallToTagged(&I, FunctionTags::HexInteger)
          || isCallToTagged(&I, FunctionTags::CharInteger)
          || isCallToTagged(&I, FunctionTags::BoolInteger)) {
        // The OperandTaintSet is discarded here. This is not a problem,
        // because it should always be empty.
        revng_assert(not hasSideEffects(I));
        revng_assert(OperandTaintSet.empty());
        continue;
      }

      // Assign instructions that have side effects
      if (hasSideEffects(I)) {
        Assignments[&I].set(Reasons::HasSideEffects);
        revng_log(MarkLog, "Instr HasSideEffects");
      }

      // For custom opcodes that don't have reference semantics there are cases
      // where we always want so serialize the instruction for aesthetic
      // reasons.
      if (not isCallToTagged(&I, FunctionTags::IsRef)) {
        // There are 2 reasons why we'd like to force serialization
        // - the instruction has zero uses and no side effects: we want to do it
        //   for debug purposes so that it shows up in the decompiled code even
        //   if it's dead
        // - the variable needs a top scope declaration. this is actually a
        //   workaround we've put in place because of the limitation of the LLVM
        //   IR, whose dominance relationships does not reflect the scoping we
        //   have in C. This can be dropped whenever we switch to a MLIR based
        //   on nested scopes
        if (not I.getNumUses() or needsTopScopeDeclaration(I)) {
          if (not I.getType()->isVoidTy()) {
            Assignments[&I].set(Reasons::AlwaysAssign);
            revng_log(MarkLog, "Instr AlwaysAssign");
          }
        }
      } else {
        revng_assert(not Assignments.contains(&I));
      }

      if (Assignments.contains(&I) or I.getType()->isVoidTy()) {

        // If we've decided to assign I, we need to consider if it might
        // interfere with other instructions that are still pending.
        if (Assignments[&I].isSet(Reasons::HasSideEffects)) {
          // If the current instruction has side effects, we also have to assign
          // all the instructions that are still pending and have interfering
          // side effects.
          revng_log(MarkLog, "Assign Pending");

          for (auto PendingIt = Pending.begin(); PendingIt != Pending.end();) {
            const auto [PendingInstr, TaintSet] = *PendingIt;
            revng_log(MarkLog,
                      "Pending: '" << PendingInstr
                                   << "': " << dumpToString(PendingInstr));
            if (haveInterferingSideEffects(&I, *PendingInstr, TaintSet)) {
              Assignments[PendingInstr].set(Reasons::HasInterferingSideEffects);
              revng_log(MarkLog, "HasInterferingSideEffects");
              PendingIt = Pending.erase(PendingIt);
            } else {
              ++PendingIt;
            }
          }
        }
      } else {
        // I is not assigned and it's not void (which are always emitted),
        // so we have to track that it's pending.
        if (not I.getType()->isVoidTy()) {
          Pending.insertWithTaint(&I, std::move(OperandTaintSet));
          revng_log(MarkLog,
                    "Add to pending: '" << &I << "': " << dumpToString(&I));
        } else {
          // The OperandTaintSet is discarded here. This is not a problem,
          // because one of the two following cases is always true.
          // - The instructions in the taint set were only ever affecting
          // the
          //   current I. Discarding them means losing track of them, but
          //   given that the void instructions are always serialized in C,
          //   this does not constitute a problem for side effects.
          // - The instruction in the taint set were also in the taint set
          // of
          //   some other instruction J. That could cause J to be assigned
          //   later for interfering side effects. This would still be
          //   correct.
          revng_log(MarkLog,
                    "void instruction without side effects: '"
                      << &I << "': " << dumpToString(&I));
        }
      }
    }

    return InterruptType::createInterrupt(std::move(Pending));
  }
};

AssignmentMap selectAssignments(llvm::Function &F) {
  MarkAssignments::Analysis Mark(F);
  Mark.initialize();
  Mark.run();
  return Mark.takeAssignments();
}
} // end namespace MarkAssignments
