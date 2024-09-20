#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <map>
#include <unordered_set>
#include <vector>

#include "boost/icl/interval_map.hpp"
#include "boost/icl/interval_set.hpp"
#include "boost/type_traits/is_same.hpp"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"

#include "revng/BasicAnalyses/MaterializedValue.h"
#include "revng/Lift/Lift.h"
#include "revng/Model/Architecture.h"
#include "revng/Model/Binary.h"
#include "revng/Model/RawBinaryView.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/MetaAddressRangeSet.h"
#include "revng/Support/ProgramCounterHandler.h"

// Forward declarations
namespace llvm {
class BasicBlock;
class Function;
class Instruction;
class LLVMContext;
class Module;
class SwitchInst;
class StoreInst;
class Value;
} // namespace llvm

class JumpTargetManager;
class ProgramCounterHandler;
class SummaryCallsBuilder;

template<typename Map>
auto containing(Map const &M, typename Map::key_type const &K) {
  auto It = M.upper_bound(K);
  if (It != M.begin())
    return --It;
  return M.end();
}

template<typename Map>
auto containing(Map &M, typename Map::key_type const &K) {
  auto It = M.upper_bound(K);
  if (It != M.begin())
    return --It;
  return M.end();
}

/// Transform constant writes to the PC in jumps
///
/// This pass looks for all the calls to the `ExitTB` function calls, looks for
/// the last write to the PC before them, checks if the written value is
/// statically known, and, if so, replaces it with a jump to the corresponding
/// translated code. If the write to the PC is not constant, no action is
/// performed, and the call to `ExitTB` remains there for later handling.
class TranslateDirectBranchesPass : public llvm::ModulePass {
public:
  TranslateDirectBranchesPass() :
    llvm::ModulePass(ID), JTM(nullptr), PCH(nullptr) {}

  TranslateDirectBranchesPass(JumpTargetManager *JTM);

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool runOnModule(llvm::Module &M) override;

private:
  /// Remove all the constant writes to the PC
  bool pinConstantStore(llvm::Function &F);

  /// Pin PC-stores for which ValueMaterializer provided useful results
  bool pinMaterializedValues(llvm::Function &F);

  /// Introduces a fallthrough branch if there's no store to PC before the last
  /// call to an helper
  ///
  /// \return true if the \p Call has been handled (i.e. a fallthrough jump has
  ///         been inserted.
  bool forceFallthroughAfterHelper(llvm::CallInst *Call);

  void pinExitTB(llvm::CallInst *ExitTBCall,
                 ProgramCounterHandler::DispatcherTargets &Destinations);

  void pinConstantStoreInternal(MetaAddress Address,
                                llvm::CallInst *ExitTBCall);

public:
  static char ID;

private:
  JumpTargetManager *JTM = nullptr;
  ProgramCounterHandler *PCH = nullptr;
};

namespace CFGForm {

/// Possible forms the CFG we're building can assume.
///
/// Generally the CFG should stay in the SemanticPreserving state, but it
/// can be temporarily changed to make certain analysis (e.g., computation of
/// the dominator tree) more effective for certain purposes.
enum Values {
  /// The CFG is an unknown state
  UnknownForm,

  /// The dispatcher jumps to all the jump targets, and all the indirect jumps
  /// go to the dispatcher
  SemanticPreserving,

  /// The dispatcher only jumps to jump targets without other predecessors and
  /// indirect jumps do not go to the dispatcher, but to an unreachable
  /// instruction
  RecoveredOnly,

  /// Similar to RecoveredOnly, but all jumps forming a function call are
  /// converted to jumps to the return address
  NoFunctionCalls
};

inline const char *getName(Values V) {
  switch (V) {
  case UnknownForm:
    return "UnknownForm";
  case SemanticPreserving:
    return "SemanticPreserving";
  case RecoveredOnly:
    return "RecoveredOnly";
  case NoFunctionCalls:
    return "NoFunctionCalls";
  }

  revng_abort();
}

} // namespace CFGForm

class CPUStateAccessAnalysisPass;

class JumpTargetManager {
private:
  using interval_set = boost::icl::interval_set<MetaAddress, CompareAddress>;
  using interval = boost::icl::interval<MetaAddress, CompareAddress>;
  using MetaAddressSet = std::unordered_set<MetaAddress>;
  using GlobalToAllocaTy = llvm::DenseMap<llvm::GlobalVariable *,
                                          llvm::AllocaInst *>;

public:
  using BlockWithAddress = std::pair<MetaAddress, llvm::BasicBlock *>;
  static const BlockWithAddress NoMoreTargets;

  class JumpTarget {
  public:
    JumpTarget() : BB(nullptr), Reasons(0) {}
    JumpTarget(llvm::BasicBlock *BB) : BB(BB), Reasons(0) {}
    JumpTarget(llvm::BasicBlock *BB, JTReason::Values Reason) :
      BB(BB), Reasons(static_cast<uint32_t>(Reason)) {}

    llvm::BasicBlock *head() const { return BB; }
    bool hasReason(JTReason::Values Reason) const {
      return (Reasons & static_cast<uint32_t>(Reason)) != 0;
    }
    void setReason(JTReason::Values Reason) {
      Reasons |= static_cast<uint32_t>(Reason);
    }
    uint32_t getReasons() const { return Reasons; }

    bool isOnlyReason(JTReason::Values Reason, JTReason::Values Ignore) const {
      return (hasReason(Reason)
              and (Reasons & ~static_cast<uint32_t>(Reason | Ignore)) == 0);
    }

    std::vector<const char *> getReasonNames() const {
      std::vector<const char *> Result;

      uint32_t LastReason = static_cast<uint32_t>(JTReason::LastReason);
      for (unsigned Reason = 1; Reason <= LastReason; Reason <<= 1) {
        JTReason::Values R = static_cast<JTReason::Values>(Reason);
        if (hasReason(R))
          Result.push_back(JTReason::getName(R));
      }

      return Result;
    }

    std::string describe() const {
      std::stringstream SS;
      SS << getName(BB) << ":";

      for (const char *ReasonName : getReasonNames())
        SS << " " << ReasonName;

      return SS.str();
    }

  private:
    llvm::BasicBlock *BB;
    uint32_t Reasons;
  };

public:
  using BlockMap = std::map<MetaAddress, JumpTarget>;
  using CSAAFactory = std::function<CPUStateAccessAnalysisPass *(void)>;

public:
  /// \param TheFunction the translated function.
  /// \param PCH ProgramCounterHandler instance.
  /// \param CreateCSAA a factory function able to create
  ///        CPUStateAccessAnalysisPass.
  JumpTargetManager(llvm::Function *TheFunction,
                    ProgramCounterHandler *PCH,
                    CSAAFactory CreateCSAA,
                    const TupleTree<model::Binary> &Model,
                    const RawBinaryView &BinaryView);

  /// Transform the IR to represent the request form of CFG
  void setCFGForm(CFGForm::Values NewForm,
                  MetaAddressSet *JumpTargetsWhitelist = nullptr);

  CFGForm::Values cfgForm() const { return CurrentCFGForm; }

  /// Collect jump targets from the program's segments
  void harvestGlobalData();

  auto createCSAA() { return CreateCSAA(); }

  /// Handle a new program counter. We might already have a basic block for that
  /// program counter, or we could even have a translation for it. Return one
  /// of these, if appropriate.
  ///
  /// \param PC the new program counter.
  /// \param ShouldContinue an out parameter indicating whether the returned
  ///        basic block was just a placeholder or actually contains a
  ///        translation.
  ///
  /// \return the basic block to use from now on, or `nullptr` if the program
  ///         counter is not associated to a basic block.
  // TODO: return pair
  llvm::BasicBlock *newPC(MetaAddress PC, bool &ShouldContinue);

  /// Save the PC-Instruction association for future use
  void registerInstruction(MetaAddress PC, llvm::Instruction *Instruction);

  auto &module() { return TheModule; }
  const auto &model() const { return Model; }

  /// Return a pointer to the `exitTB` function
  ///
  /// `exitTB` is called when jump to the current value of the PC must be
  /// performed.
  llvm::Function *exitTB() { return ExitTB; }

  /// Pop from the list of program counters to explore
  ///
  /// \return a pair containing the PC and the initial block to use, or
  ///         JumpTarget::NoMoreTargets if we're done.
  BlockWithAddress peek();

  /// Return true if no unexplored jump targets are available
  bool empty() { return Unexplored.empty(); }

  /// Return true if the whole [\p Start,\p End) range is in an executable
  /// segment
  bool isExecutableRange(const MetaAddress &Start,
                         const MetaAddress &End) const {
    return ExecutableRanges.contains(Start, End);
  }

  bool isMapped(MetaAddress Start, MetaAddress End) const {
    revng_assert(Start.isValid() and End.isValid());

    for (const model::Segment &Segment : Model->Segments()) {
      if (Segment.StartAddress().addressLowerThanOrEqual(Start)
          and Start.addressLowerThan(Segment.endAddress())
          and Segment.StartAddress().addressLowerThanOrEqual(End)
          and End.addressLowerThan(Segment.endAddress())) {
        return true;
      }
    }

    return false;
  }

  /// Return true if the given PC can be executed by the current architecture
  bool isPC(MetaAddress PC) const {
    revng_assert(PC.isValid());
    return isExecutableAddress(PC);
  }

  /// Return true if the given PC is a jump target
  bool isJumpTarget(MetaAddress PC) const {
    revng_assert(PC.isValid());
    return JumpTargets.contains(PC);
  }

  /// Return true if the given basic block corresponds to a jump target
  bool isJumpTarget(llvm::BasicBlock *BB) {
    if (BB->empty())
      return false;

    MetaAddress PC = getBasicBlockAddress(BB);
    if (PC.isValid())
      return isJumpTarget(PC);

    return false;
  }

  /// Return true if \p PC is in an executable segment
  bool isExecutableAddress(const MetaAddress &PC) const {
    return ExecutableRanges.contains(PC);
  }

  /// Get the basic block associated to the original address \p PC
  ///
  /// If the given address has never been met, assert.
  ///
  /// \param PC the PC for which a `BasicBlock` is requested.
  llvm::BasicBlock *getBlockAt(MetaAddress PC);

  /// Return, and, if necessary, register the basic block associated to \p PC
  ///
  /// This function can return `nullptr`.
  ///
  /// \param PC the PC for which a `BasicBlock` is requested.
  ///
  /// \return a `BasicBlock`, it might be newly created and empty, empty and
  ///         created in the past or even a `BasicBlock` already containing the
  ///         translated code.  It might also return `nullptr` if the PC is not
  ///         valid or another error occurred.
  llvm::BasicBlock *registerJT(MetaAddress PC, JTReason::Values Reason);

  bool hasJT(MetaAddress PC) {
    revng_assert(PC.isValid());
    return JumpTargets.contains(PC);
  }

  BlockMap::const_iterator begin() const { return JumpTargets.begin(); }

  BlockMap::const_iterator end() const { return JumpTargets.end(); }

  void registerJT(llvm::BasicBlock *BB, JTReason::Values Reason) {
    registerJT(getBasicBlockAddress(notNull(BB)), Reason);
  }

  // TODO: this is a likely approach is broken, it depends on the order
  /// As registerJT, but only if the JT has already been registered
  ///
  /// \return true if the given PC did not already have such reason
  bool markJT(MetaAddress PC, JTReason::Values Reason) {
    bool Result = false;
    revng_assert(PC.isValid());

    if (isJumpTarget(PC)) {
      Result = not JumpTargets.at(PC).hasReason(Reason);
      registerJT(PC, Reason);
    }

    return Result;
  }

  /// Checks if \p BB is a basic block generated during translation
  bool isTranslatedBB(llvm::BasicBlock *BB) const {
    return BB != anyPC() && BB != unexpectedPC() && BB != dispatcher()
           && BB != dispatcherFail();
  }

  /// Return the dispatcher basic block.
  ///
  /// \note Do not use this for comparison with successors of translated code,
  ///       use isTranslatedBB instead.
  llvm::BasicBlock *dispatcher() const { return Dispatcher; }

  /// Return the basic block handling an unknown PC in the dispatcher
  llvm::BasicBlock *dispatcherFail() const { return DispatcherFail; }

  /// Return the basic block handling a jump to any PC
  llvm::BasicBlock *anyPC() const { return AnyPC; }

  /// Return the basic block handling a jump to an unexpected PC
  llvm::BasicBlock *unexpectedPC() const { return UnexpectedPC; }

  // TODO: can this be replaced by the corresponding method in
  // GeneratedCodeBasicInfo?
  /// Get the PC associated and the size of the original instruction
  std::pair<MetaAddress, uint64_t>
  getPC(llvm::Instruction *TheInstruction) const;

  // TODO: can this be replaced by the corresponding method in
  // GeneratedCodeBasicInfo?
  MetaAddress getNextPC(llvm::Instruction *TheInstruction) const {
    auto Pair = getPC(TheInstruction);
    return Pair.first + Pair.second;
  }

  MaterializedValue readFromPointer(MetaAddress LoadAddress,
                                    unsigned LoadSize,
                                    bool IsLittleEndian);

  /// Increment the counter of emitted branches since the last reset
  void recordNewBranches(llvm::BasicBlock *Source, size_t Count) {
    ValueMaterializerPCWhiteList.insert(getPC(Source->getTerminator()).first);
    NewBranches += Count;
  }

  bool isInValueMaterializerPCWhitelist(MetaAddress Address) const {
    return ValueMaterializerPCWhiteList.contains(Address);
  }

  void clearValueMaterializerPCWhitelist() {
    ValueMaterializerPCWhiteList.clear();
  }

  /// Finalizes information about the jump targets
  ///
  /// Call this function once no more jump targets can be discovered.  It will
  /// fix all the pending information. In particular, those pointers to code
  /// that have never been touched will be considered and their pointee will be
  /// marked with UnusedGlobalData.
  ///
  /// This function also fixes the "anypc" and "unexpectedpc" basic blocks to
  /// their proper behavior.
  void finalizeJumpTargets() {
    fixPostHelperPC();

    translateIndirectJumps();

    using namespace model::Architecture;
    uint64_t ReadSize = getPointerSize(Model->Architecture());
    for (MetaAddress MemoryAddress : UnusedCodePointers) {
      // Read using the original endianness, we want the correct address
      auto MaybeRawPC = BinaryView.readInteger(MemoryAddress, ReadSize);
      MetaAddress PC = MetaAddress::invalid();
      if (MaybeRawPC)
        PC = fromPC(*MaybeRawPC);

      if (PC.isValid()) {
        // Set as reason UnusedGlobalData and ensure it's not empty
        llvm::BasicBlock *BB = registerJT(PC, JTReason::UnusedGlobalData);

        // TODO: can this happen?
        revng_assert(BB != nullptr);

        revng_assert(!BB->empty());
      }
    }

    // We no longer need this information
    freeContainer(UnusedCodePointers);
  }

  MetaAddress fromPC(uint64_t PC) const {
    using namespace model::Architecture;
    auto Architecture = toLLVMArchitecture(Model->Architecture());
    return MetaAddress::fromPC(Architecture, PC);
  }

  MetaAddress fromGeneric(uint64_t Address) const {
    using namespace model::Architecture;
    auto Architecture = toLLVMArchitecture(Model->Architecture());
    return MetaAddress::fromGeneric(Architecture, Address);
  }

  MetaAddress fromPCStore(llvm::StoreInst *Store) {
    auto *Constant = llvm::cast<llvm::ConstantInt>(Store->getValueOperand());
    return fromPC(Constant->getLimitedValue());
  }

  void createJTReasonMD() {
    using namespace llvm;

    Function *CallMarker = TheModule.getFunction("function_call");
    if (CallMarker != nullptr) {
      auto UnwrapBA = [](Value *V) {
        return cast<BlockAddress>(V)->getBasicBlock();
      };
      for (User *U : CallMarker->users()) {
        if (CallInst *Call = dyn_cast<CallInst>(U)) {
          if (isa<BlockAddress>(Call->getOperand(0)))
            registerJT(UnwrapBA(Call->getOperand(0)), JTReason::Callee);
          registerJT(UnwrapBA(Call->getOperand(1)), JTReason::ReturnAddress);
        }
      }
    }

    // Tag each jump target with its reasons
    for (auto &P : JumpTargets) {
      JumpTarget &JT = P.second;
      Instruction *T = JT.head()->getTerminator();
      revng_assert(T != nullptr);

      std::vector<Metadata *> Reasons;
      for (const char *ReasonName : JT.getReasonNames())
        Reasons.push_back(MDString::get(Context, ReasonName));

      T->setMetadata("revng.jt.reasons", MDTuple::get(Context, Reasons));
    }
  }

  void registerReadRange(MetaAddress StartAddress, MetaAddress EndAddress);

  const interval_set &readRange() const { return ReadIntervalSet; }

  std::string nameForAddress(MetaAddress Address, uint64_t Size = 1) const {
    // TODO: we should have a Binary::nameForAddress() which uses the model to
    //      find a proper name
    return Address.toString();
  }

  /// Register a simple literal collected during translation for harvesting
  ///
  /// A simple literal is a literal value found in the input program that is
  /// simple enough not to require more sophisticated analyses. The typical
  /// example is the return address of a function call, that is provided to use
  /// by libtinycode in full.
  ///
  /// Simple literals are registered as possible jump targets before attempting
  /// more expensive techniques.
  void registerSimpleLiteral(MetaAddress Address) {
    SimpleLiterals.insert(Address);
  }

  ProgramCounterHandler *programCounterHandler() { return PCH; }

  llvm::DenseSet<llvm::BasicBlock *> computeUnreachable() const;

private:
  void fixPostHelperPC();

  /// Translate the non-constant jumps into jumps to the dispatcher
  void translateIndirectJumps();

  /// Erase \p I, and deregister it in case it's a call to `newpc`
  void eraseInstruction(llvm::Instruction *I) {
    revng_assert(I->use_empty());

    MetaAddress PC = getBasicBlockAddress(I->getParent());
    if (PC.isValid())
      OriginalInstructionAddresses.erase(PC);
    eraseFromParent(I);
  }

  /// Drop \p Start and all the descendants, stopping when a JT is met
  void purgeTranslation(llvm::BasicBlock *Start);

  /// Check if \p BB has at least a predecessor, excluding the dispatcher
  bool hasPredecessors(llvm::BasicBlock *BB) const;

  /// Rebuild the dispatcher switch
  ///
  /// Depending on the CFG form we're currently adopting the dispatcher might go
  /// to all the jump targets or only to those who have no other predecessor.
  void rebuildDispatcher(MetaAddressSet *Whitelist);

  void prepareDispatcher();

  template<typename value_type, unsigned endian>
  void findCodePointers(MetaAddress StartVirtualAddress,
                        const unsigned char *Start,
                        const unsigned char *End);

  void harvest();

  llvm::CallInst *getJumpTarget(llvm::BasicBlock *Target);

private:
  using InstructionMap = std::map<MetaAddress, llvm::Instruction *>;

  llvm::Module &TheModule;
  llvm::LLVMContext &Context;
  llvm::Function *TheFunction;
  /// Holds the association between a PC and the last generated instruction for
  /// the previous instruction.
  InstructionMap OriginalInstructionAddresses;
  /// Holds the association between a PC and a BasicBlock.
  BlockMap JumpTargets;
  /// Queue of program counters we still have to translate.
  std::vector<BlockWithAddress> Unexplored;

  llvm::Function *ExitTB;
  MetaAddressRangeSet ExecutableRanges;

  llvm::BasicBlock *Dispatcher;
  llvm::SwitchInst *DispatcherSwitch;
  llvm::BasicBlock *DispatcherFail;
  llvm::BasicBlock *AnyPC;
  llvm::BasicBlock *UnexpectedPC;

  unsigned NewBranches = 0;

  std::set<MetaAddress> UnusedCodePointers;
  interval_set ReadIntervalSet;

  CFGForm::Values CurrentCFGForm;
  std::set<llvm::BasicBlock *> ToPurge;
  std::set<MetaAddress> SimpleLiterals;
  CSAAFactory CreateCSAA;

  ProgramCounterHandler *PCH = nullptr;

  MetaAddressSet ValueMaterializerPCWhiteList;
  const TupleTree<model::Binary> &Model;
  const RawBinaryView &BinaryView;
  bool AftedAddingFunctionEntries = false;
};

template<>
struct BlackListTrait<const JumpTargetManager &, llvm::BasicBlock *>
  : BlackListTraitBase<const JumpTargetManager &> {
  using BlackListTraitBase<const JumpTargetManager &>::BlackListTraitBase;
  bool isBlacklisted(llvm::BasicBlock *Value) {
    return !this->Obj.isTranslatedBB(Value);
  }
};

inline BlackListTrait<const JumpTargetManager &, llvm::BasicBlock *>
make_blacklist(const JumpTargetManager &JTM) {
  return BlackListTrait<const JumpTargetManager &, llvm::BasicBlock *>(JTM);
}
