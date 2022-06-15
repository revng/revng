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
#include "revng/Support/MonotoneFramework.h"

#include "revng-c/Backend/VariableScopeAnalysis.h"
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

static bool hasSideEffects(const llvm::Instruction &I) {
  if (isa<llvm::StoreInst>(&I))
    return true;

  if (auto *Call = llvm::dyn_cast<llvm::CallInst>(&I)) {
    auto *CalledFunc = Call->getCalledFunction();

    if (not CalledFunc)
      return false;

    if (FunctionTags::Isolated.isTagOf(CalledFunc)
        or FunctionTags::WritesMemory.isTagOf(CalledFunc)
        or FunctionTags::QEMU.isTagOf(CalledFunc) or CalledFunc->isIntrinsic())
      return true;
  }

  return false;
}

using TaintSetT = std::set<const llvm::Instruction *>;

static bool
haveInterferingSideEffects(const llvm::Instruction & /*InstrWithSideEffects*/,
                           const llvm::Instruction &Other,
                           const TaintSetT &TaintSet) {

  auto MightInterfere = [](const llvm::Instruction *I) {
    if (isa<llvm::StoreInst>(I) or isa<llvm::LoadInst>(I))
      return true;

    if (auto *Call = llvm::dyn_cast<llvm::CallInst>(I)) {
      auto *CalledFunc = Call->getCalledFunction();

      if (not CalledFunc)
        return false;

      if (FunctionTags::Isolated.isTagOf(CalledFunc)
          or FunctionTags::WritesMemory.isTagOf(CalledFunc)
          or FunctionTags::ReadsMemory.isTagOf(CalledFunc)
          or FunctionTags::QEMU.isTagOf(CalledFunc)
          or CalledFunc->isIntrinsic())
        return true;
    }

    return false;
  };

  // TODO: at the moment we consider any instruction that might interfere with
  // an instruction with a side effect as interfering. We might want to refine
  // this by verifying if the interference actually occurs.
  return MightInterfere(&Other) or llvm::any_of(TaintSet, MightInterfere);
}

class IntersectionMonotoneSetWithTaint;

using LatticeElement = IntersectionMonotoneSetWithTaint;

class IntersectionMonotoneSetWithTaint {
public:
  using Instruction = llvm::Instruction;
  using TaintMap = std::map<Instruction *, std::set<const Instruction *>>;
  using const_iterator = typename TaintMap::const_iterator;
  using iterator = typename TaintMap::iterator;
  using size_type = typename TaintMap::size_type;

protected:
  TaintMap TaintedPending;
  bool IsBottom;

protected:
  IntersectionMonotoneSetWithTaint(const IntersectionMonotoneSetWithTaint &) =
    default;

public:
  IntersectionMonotoneSetWithTaint() : TaintedPending(), IsBottom(true){};

  IntersectionMonotoneSetWithTaint copy() const { return *this; }
  IntersectionMonotoneSetWithTaint &
  operator=(const IntersectionMonotoneSetWithTaint &) = default;

  IntersectionMonotoneSetWithTaint(IntersectionMonotoneSetWithTaint &&) =
    default;
  IntersectionMonotoneSetWithTaint &
  operator=(IntersectionMonotoneSetWithTaint &&) = default;

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

  void insert(Instruction *Key) {
    revng_assert(not IsBottom);
    TaintedPending[Key];
  }

  size_type erase(Instruction *El) {
    revng_assert(not IsBottom);
    return TaintedPending.erase(El);
  }

  const_iterator erase(const_iterator It) {
    revng_assert(not IsBottom);
    return this->TaintedPending.erase(It);
  }

  bool isPending(Instruction *Key) const {
    revng_assert(not IsBottom);
    return TaintedPending.count(Key);
  }

public:
  static IntersectionMonotoneSetWithTaint bottom() {
    return IntersectionMonotoneSetWithTaint();
  }

  static IntersectionMonotoneSetWithTaint top() {
    IntersectionMonotoneSetWithTaint Res = {};
    Res.IsBottom = false;
    return Res;
  }

public:
  void combine(const IntersectionMonotoneSetWithTaint &Other) {
    // Simply intersects the sets
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

    iterator SetIt = this->TaintedPending.begin();
    iterator SetEnd = this->TaintedPending.end();
    for (; SetIt != SetEnd; ++SetIt) {
      iterator OtherIt = OtherCopy.TaintedPending.find(SetIt->first);
      if (OtherIt == OtherEnd)
        ToDrop.push_back(SetIt);
      else
        SetIt->second.merge(OtherIt->second);
    }

    for (iterator I : ToDrop)
      this->TaintedPending.erase(I);
  }

  bool lowerThanOrEqual(const IntersectionMonotoneSetWithTaint &Other) const {
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

  llvm::Optional<LatticeElement>
  handleEdge(const LatticeElement &Original,
             const llvm::BasicBlock * /*Source*/,
             const llvm::BasicBlock *Destination) const {

    auto LiveInIt = LiveIn.find(Destination);
    if (LiveInIt == LiveIn.end())
      return llvm::None;

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

      {
        // Operands are removed from pending
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
              Pending.erase(UsedInstr);
            } else {
              revng_log(MarkLog, "Op is NOT Instruction: leave it in pending");
              revng_assert(isa<Argument>(V) or isa<Constant>(V)
                           or isa<BasicBlock>(V) or isa<MetadataAsValue>(V));
            }
          }
        }
      }

      // After the new redesign of IRCanonicalization PHINodes shouldn't even
      // reach this stage.
      revng_assert(not isa<PHINode>(I));

      // Skip branching instructions, since they never generate assignments.
      if (isa<BranchInst>(I) or isa<SwitchInst>(I))
        continue;

      if (hasSideEffects(I)) {
        Assignments[&I].set(Reasons::HasSideEffects);
        revng_log(MarkLog, "Instr HasSideEffects");
      }

      switch (I.getNumUses()) {

      case 1: {
        // Instructions with a single used do not necessarily need to generate
        // an assignment.
      } break;

      case 0: {
        // Force unused instructions to be assigned. This is done to ease
        // debugging, and could potentially be dropped in the future.
        if (not I.getType()->isVoidTy()) {
          Assignments[&I].set(Reasons::AlwaysAssign);
          revng_log(MarkLog, "Instr AlwaysAssign");
        }
      } break;

      default: {
        // Instructions with more than one use are always assigned, so all the
        // users can re-use the assigned variable.
        Assignments[&I].set(Reasons::HasManyUses);
        revng_log(MarkLog, "Instr HasManyUses: " << I.getNumUses());
      } break;
      }

      // If an instruction is used outside of the scope in which it appears in
      // the LLVM IR, we need to create a local variable for it.
      if (needsTopScopeDeclaration(I)) {
        Assignments[&I].set(Reasons::HasUsesOutsideBB);
        revng_log(MarkLog,
                  "Instr has uses outside its basic block: " << I.getNumUses());
      }

      auto SerIt = Assignments.find(&I);
      if (SerIt != Assignments.end()
          and (SerIt->second.hasSideEffects()
               or SerIt->second.isSet(Reasons::AlwaysAssign))) {
        revng_log(MarkLog, "Assign Pending");
        // We also have to assign all the instructions that are still pending
        // and have interfering side effects.
        for (auto PendingIt = Pending.begin(); PendingIt != Pending.end();) {
          auto *PendingInstr = PendingIt->first;
          revng_log(MarkLog,
                    "Pending: '" << PendingInstr
                                 << "': " << dumpToString(PendingInstr));
          if (haveInterferingSideEffects(I, *PendingInstr, PendingIt->second)) {
            Assignments[PendingInstr].set(Reasons::HasInterferingSideEffects);
            revng_log(MarkLog, "HasInterferingSideEffects");

            PendingIt = Pending.erase(PendingIt);
          } else {
            ++PendingIt;
          }
        }
      } else {
        Pending.insert(&I);
        revng_log(MarkLog,
                  "Add to pending: '" << &I << "': " << dumpToString(&I));
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
