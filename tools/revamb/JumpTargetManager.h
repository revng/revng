#ifndef JUMPTARGETMANAGER_H
#define JUMPTARGETMANAGER_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdint>
#include <map>
#include <set>
#include <vector>

// Boost includes
#include <boost/icl/interval_map.hpp>
#include <boost/icl/interval_set.hpp>
#include <boost/type_traits/is_same.hpp>

// LLVM includes
#include "llvm/ADT/Optional.h"
#include "llvm/IR/Instructions.h"

// Local libraries includes
#include "revng/Support/IRHelpers.h"
#include "revng/Support/revng.h"

// Local includes
#include "BinaryFile.h"
#include "NoReturnAnalysis.h"

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

template<typename Map>
typename Map::const_iterator
containing(Map const &m, typename Map::key_type const &k) {
  typename Map::const_iterator it = m.upper_bound(k);
  if (it != m.begin()) {
    return --it;
  }
  return m.end();
}

template<typename Map>
typename Map::iterator containing(Map &m, typename Map::key_type const &k) {
  typename Map::iterator it = m.upper_bound(k);
  if (it != m.begin()) {
    return --it;
  }
  return m.end();
}

/// \brief Transform constant writes to the PC in jumps
///
/// This pass looks for all the calls to the `ExitTB` function calls, looks for
/// the last write to the PC before them, checks if the written value is
/// statically known, and, if so, replaces it with a jump to the corresponding
/// translated code. If the write to the PC is not constant, no action is
/// performed, and the call to `ExitTB` remains there for later handling.
class TranslateDirectBranchesPass : public llvm::FunctionPass {
public:
  static char ID;

  TranslateDirectBranchesPass() : llvm::FunctionPass(ID), JTM(nullptr) {}

  TranslateDirectBranchesPass(JumpTargetManager *JTM) :
    FunctionPass(ID),
    JTM(JTM) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool runOnFunction(llvm::Function &F) override;

  /// \brief Remove all the constant writes to the PC
  bool pinConstantStore(llvm::Function &F);

  /// \brief Remove all the PC-writes for which a set of (approximate) targets
  ///        is known
  bool pinJTs(llvm::Function &F);

  /// Introduces a fallthrough branch if there's no store to PC before the last
  /// call to an helper
  ///
  /// \return true if the \p Call has been handled (i.e. a fallthrough jump has
  ///         been inserted.
  bool forceFallthroughAfterHelper(llvm::CallInst *Call);

private:
  /// Obtains the absolute address of the PC corresponding to the original
  /// assembly instruction coming after the specified LLVM instruction
  uint64_t getNextPC(llvm::Instruction *TheInstruction);

private:
  JumpTargetManager *JTM;
};

namespace CFGForm {

/// \brief Possible forms the CFG we're building can assume.
///
/// Generally the CFG should stay in the SemanticPreservingCFG state, but it
/// can be temporarily changed to make certain analysis (e.g., computation of
/// the dominator tree) more effective for certain purposes.
enum Values {
  /// The CFG is an unknown state
  UnknownFormCFG,
  /// The dispatcher jumps to all the jump targets, and all the indirect jumps
  /// go to the dispatcher
  SemanticPreservingCFG,
  /// The dispatcher only jumps to jump targets without other predecessors and
  /// indirect jumps do not go to the dispatcher, but to an unreachable
  /// instruction
  RecoveredOnlyCFG,
  /// Similar to RecoveredOnlyCFG, but all jumps forming a function call are
  /// converted to jumps to the return address
  NoFunctionCallsCFG
};

inline const char *getName(Values V) {
  switch (V) {
  case UnknownFormCFG:
    return "UnknownFormCFG";
  case SemanticPreservingCFG:
    return "SemanticPreservingCFG";
  case RecoveredOnlyCFG:
    return "RecoveredOnlyCFG";
  case NoFunctionCallsCFG:
    return "NoFunctionCallsCFG";
  }

  revng_abort();
}

} // namespace CFGForm

class JumpTargetManager {
private:
  using interval_set = boost::icl::interval_set<uint64_t>;
  using interval = boost::icl::interval<uint64_t>;

public:
  using BlockWithAddress = std::pair<uint64_t, llvm::BasicBlock *>;
  static const BlockWithAddress NoMoreTargets;

  class JumpTarget {
  public:
    JumpTarget() : BB(nullptr), Reasons(0) {}
    JumpTarget(llvm::BasicBlock *BB) : BB(BB), Reasons(0) {}
    JumpTarget(llvm::BasicBlock *BB, JTReason::Values Reason) :
      BB(BB),
      Reasons(static_cast<uint32_t>(Reason)) {}

    llvm::BasicBlock *head() const { return BB; }
    bool hasReason(JTReason::Values Reason) const {
      return (Reasons & static_cast<uint32_t>(Reason)) != 0;
    }
    void setReason(JTReason::Values Reason) {
      Reasons |= static_cast<uint32_t>(Reason);
    }
    uint32_t getReasons() const { return Reasons; }

    bool isOnlyReason(JTReason::Values Reason) const {
      return (hasReason(Reason)
              and (Reasons & ~static_cast<uint32_t>(Reason)) == 0);
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
  using RangesVector = std::vector<std::pair<uint64_t, uint64_t>>;

  /// \param TheFunction the translated function.
  /// \param PCReg the global variable representing the program counter.
  /// \param Binary reference to the information about a given binary, such as
  ///        segments and symbols.
  JumpTargetManager(llvm::Function *TheFunction,
                    llvm::Value *PCReg,
                    const BinaryFile &Binary);

  /// \brief Transform the IR to represent the request form of CFG
  void setCFGForm(CFGForm::Values NewForm);

  CFGForm::Values cfgForm() const { return CurrentCFGForm; }

  /// \brief Collect jump targets from the program's segments
  void harvestGlobalData();

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
  llvm::BasicBlock *newPC(uint64_t PC, bool &ShouldContinue);

  /// \brief Save the PC-Instruction association for future use
  void registerInstruction(uint64_t PC, llvm::Instruction *Instruction);

  /// \brief Return the most recent instruction writing the program counter
  ///
  /// Note that the search is performed only in the current basic block.  The
  /// function will assert if the write instruction is not found.
  ///
  /// \param TheInstruction instruction from which start the search.
  ///
  /// \return a pointer to the last `StoreInst` writing the program counter, or
  ///         `nullptr` if a call to an helper has been found before the write
  ///         to the PC.
  llvm::StoreInst *getPrevPCWrite(llvm::Instruction *TheInstruction);

  /// \brief Return a pointer to the `exitTB` function
  ///
  /// `exitTB` is called when jump to the current value of the PC must be
  /// performed.
  llvm::Function *exitTB() { return ExitTB; }

  /// \brief Pop from the list of program counters to explore
  ///
  /// \return a pair containing the PC and the initial block to use, or
  ///         JumpTarget::NoMoreTargets if we're done.
  BlockWithAddress peek();

  /// \brief Return true if no unexplored jump targets are available
  bool empty() { return Unexplored.empty(); }

  /// \brief Return true if the whole [\p Start,\p End) range is in an
  ///        executable segment
  bool isExecutableRange(uint64_t Start, uint64_t End) const {
    for (std::pair<uint64_t, uint64_t> Range : ExecutableRanges)
      if (Range.first <= Start && Start < Range.second && Range.first <= End
          && End < Range.second)
        return true;
    return false;
  }

  /// \brief Return true if the given PC respects the input architecture's
  ///        instruction alignment constraints
  bool isInstructionAligned(uint64_t PC) const {
    return PC % Binary.architecture().instructionAlignment() == 0;
  }

  /// \brief Return true if the given PC can be executed by the current
  ///        architecture
  bool isPC(uint64_t PC) const {
    return isExecutableAddress(PC) && isInstructionAligned(PC);
  }

  /// \brief Return true if the given PC is a jump target
  bool isJumpTarget(uint64_t PC) const { return JumpTargets.count(PC); }

  /// \brief Return true if the given basic block corresponds to a jump target
  bool isJumpTarget(llvm::BasicBlock *BB) {
    if (BB->empty())
      return false;

    uint64_t PC = getPCFromNewPCCall(&*BB->begin());
    if (PC != 0)
      return isJumpTarget(PC);

    return false;
  }

  /// \brief Return true if \p PC is in an executable segment
  bool isExecutableAddress(uint64_t PC) const {
    for (std::pair<uint64_t, uint64_t> Range : ExecutableRanges)
      if (Range.first <= PC && PC < Range.second)
        return true;
    return false;
  }

  /// \brief Get the basic block associated to the original address \p PC
  ///
  /// If the given address has never been met, assert.
  ///
  /// \param PC the PC for which a `BasicBlock` is requested.
  llvm::BasicBlock *getBlockAt(uint64_t PC);

  /// \brief Return, and, if necessary, register the basic block associated to
  ///        \p PC
  ///
  /// This function can return `nullptr`.
  ///
  /// \param PC the PC for which a `BasicBlock` is requested.
  ///
  /// \return a `BasicBlock`, it might be newly created and empty, empty and
  ///         created in the past or even a `BasicBlock` already containing the
  ///         translated code.  It might also return `nullptr` if the PC is not
  ///         valid or another error occurred.
  llvm::BasicBlock *registerJT(uint64_t PC, JTReason::Values Reason);

  bool hasJT(uint64_t PC) { return JumpTargets.count(PC) != 0; }

  std::map<uint64_t, JumpTarget>::const_iterator begin() const {
    return JumpTargets.begin();
  }

  std::map<uint64_t, JumpTarget>::const_iterator end() const {
    return JumpTargets.end();
  }

  void registerJT(llvm::BasicBlock *BB, JTReason::Values Reason) {
    revng_assert(!BB->empty());
    auto *CallNewPC = llvm::dyn_cast<llvm::CallInst>(&*BB->begin());
    revng_assert(CallNewPC != nullptr);
    llvm::Function *Callee = CallNewPC->getCalledFunction();
    revng_assert(Callee != nullptr && Callee->getName() == "newpc");
    registerJT(getLimitedValue(CallNewPC->getArgOperand(0)), Reason);
  }

  /// \brief As registerJT, but only if the JT has already been registered
  void markJT(uint64_t PC, JTReason::Values Reason) {
    if (isJumpTarget(PC))
      registerJT(PC, Reason);
  }

  /// \brief Removes a `BasicBlock` from the SET's visited list
  void unvisit(llvm::BasicBlock *BB);

  /// \brief Checks if \p BB is a basic block generated during translation
  bool isTranslatedBB(llvm::BasicBlock *BB) const {
    return BB != anyPC() && BB != unexpectedPC() && BB != dispatcher()
           && BB != dispatcherFail();
  }

  /// \brief Return the dispatcher basic block.
  ///
  /// \note Do not use this for comparison with successors of translated code,
  ///       use isTranslatedBB instead.
  llvm::BasicBlock *dispatcher() const { return Dispatcher; }

  /// \brief Return the basic block handling an unknown PC in the dispatcher
  llvm::BasicBlock *dispatcherFail() const { return DispatcherFail; }

  /// \brief Return the basic block handling a jump to any PC
  llvm::BasicBlock *anyPC() const { return AnyPC; }

  /// \brief Return the basic block handling a jump to an unexpected PC
  llvm::BasicBlock *unexpectedPC() const { return UnexpectedPC; }

  bool isPCReg(llvm::Value *TheValue) const { return TheValue == PCReg; }

  llvm::Value *pcReg() const { return PCReg; }

  // TODO: can this be replaced by the corresponding method in
  // GeneratedCodeBasicInfo?
  /// \brief Get the PC associated to \p TheInstruction and the next one
  ///
  /// \return a pair containing the PC associated to \p TheInstruction and the
  ///         next one.
  std::pair<uint64_t, uint64_t> getPC(llvm::Instruction *TheInstruction) const;

  // TODO: can this be replaced by the corresponding method in
  // GeneratedCodeBasicInfo?
  uint64_t getNextPC(llvm::Instruction *TheInstruction) const {
    auto Pair = getPC(TheInstruction);
    return Pair.first + Pair.second;
  }

  /// \brief Read an integer number from a segment
  ///
  /// \param Address the address from which to read.
  /// \param Size the size of the read in bytes.
  ///
  /// \return a `ConstantInt` with the read value or `nullptr` in case it wasn't
  ///         possible to read the value (e.g., \p Address is not inside any of
  ///         the segments).
  llvm::ConstantInt *
  readConstantInt(llvm::Constant *Address,
                  unsigned Size,
                  BinaryFile::Endianess E = BinaryFile::OriginalEndianess);

  /// \brief Reads a pointer-sized value from a segment
  /// \see readConstantInt
  llvm::Constant *
  readConstantPointer(llvm::Constant *Address,
                      llvm::Type *PointerTy,
                      BinaryFile::Endianess E = BinaryFile::OriginalEndianess);

  /// \brief Increment the counter of emitted branches since the last reset
  void newBranch() { NewBranches++; }

  /// \brief Finalizes information about the jump targets
  ///
  /// Call this function once no more jump targets can be discovered.  It will
  /// fix all the pending information. In particular, those pointers to code
  /// that have never been touched by SET will be considered and their pointee
  /// will be marked with UnusedGlobalData.
  ///
  /// This function also fixes the "anypc" and "unexpectedpc" basic blocks to
  /// their proper behavior.
  void finalizeJumpTargets() {
    translateIndirectJumps();

    unsigned ReadSize = Binary.architecture().pointerSize() / 8;
    for (uint64_t MemoryAddress : UnusedCodePointers) {
      // Read using the original endianess, we want the correct address
      uint64_t PC = *Binary.readRawValue(MemoryAddress, ReadSize);

      // Set as reason UnusedGlobalData and ensure it's not empty
      llvm::BasicBlock *BB = registerJT(PC, JTReason::UnusedGlobalData);
      revng_assert(!BB->empty());
    }

    // We no longer need this information
    freeContainer(UnusedCodePointers);
  }

  void createJTReasonMD() {
    using namespace llvm;

    Function *CallMarker = TheModule.getFunction("function_call");
    if (CallMarker != nullptr) {
      auto unwrapBA = [](Value *V) {
        return cast<BlockAddress>(V)->getBasicBlock();
      };
      for (User *U : CallMarker->users()) {
        if (CallInst *Call = dyn_cast<CallInst>(U)) {
          if (isa<BlockAddress>(Call->getOperand(0)))
            registerJT(unwrapBA(Call->getOperand(0)), JTReason::Callee);
          registerJT(unwrapBA(Call->getOperand(1)), JTReason::ReturnAddress);
        }
      }
    }

    // Tag each jump target with its reasons
    for (auto &P : JumpTargets) {
      JumpTarget &JT = P.second;
      TerminatorInst *T = JT.head()->getTerminator();
      revng_assert(T != nullptr);

      std::vector<Metadata *> Reasons;
      for (const char *ReasonName : JT.getReasonNames())
        Reasons.push_back(MDString::get(Context, ReasonName));

      T->setMetadata("revamb.jt.reasons", MDTuple::get(Context, Reasons));
    }
  }

  unsigned delaySlotSize() const {
    return Binary.architecture().delaySlotSize();
  }

  const BinaryFile &binary() const { return Binary; }

  /// \brief Return the next call to exitTB after I, or nullptr if it can't find
  ///        one
  llvm::CallInst *findNextExitTB(llvm::Instruction *I);

  // TODO: can we drop this in favor of GeneratedCodeBasicInfo::isJump?
  bool isJump(llvm::TerminatorInst *T) const {
    for (llvm::BasicBlock *Successor : T->successors()) {
      if (!(Successor == Dispatcher || Successor == DispatcherFail
            || isJumpTarget(getBasicBlockPC(Successor))))
        return false;
    }

    return true;
  }

  void registerReadRange(uint64_t Address, uint64_t Size);

  const interval_set &readRange() const { return ReadIntervalSet; }

  NoReturnAnalysis &noReturn() { return NoReturn; }

  /// \brief Return a proper name for the given address, possibly using symbols
  ///
  /// \param Address the address for which a name should be produced.
  ///
  /// \return a string containing the symbol name and, if necessary an offset,
  ///         or if no symbol can be found, just the address.
  std::string nameForAddress(uint64_t Address, uint64_t Size = 1) const;

  /// \brief Register a simple literal collected during translation for
  ///        harvesting
  ///
  /// A simple literal is a literal value found in the input program that is
  /// simple enough not to require SET. The typcal example is the return address
  /// of a function call, that is provided to use by libtinycode in full.
  ///
  /// Simple literals are registered as possible jump targets before attempting
  /// more expensive techniques such as SET.
  void registerSimpleLiteral(uint64_t Address) {
    SimpleLiterals.insert(Address);
  }

private:
  std::set<llvm::BasicBlock *> computeUnreachable();

  /// \brief Translate the non-constant jumps into jumps to the dispatcher
  void translateIndirectJumps();

  /// \brief Helper function to check if an instruction is a call to `newpc`
  ///
  /// \return 0 if \p I is not a call to `newpc`, otherwise the PC address of
  ///         associated to the call to `newpc`
  uint64_t getPCFromNewPCCall(llvm::Instruction *I) {
    if (auto *CallNewPC = llvm::dyn_cast<llvm::CallInst>(I)) {
      if (CallNewPC->getCalledFunction() == nullptr
          || CallNewPC->getCalledFunction()->getName() != "newpc")
        return 0;

      return getLimitedValue(CallNewPC->getArgOperand(0));
    }

    return 0;
  }

  /// \brief Erase \p I, and deregister it in case it's a call to `newpc`
  void eraseInstruction(llvm::Instruction *I) {
    revng_assert(I->use_empty());

    uint64_t PC = getPCFromNewPCCall(I);
    if (PC != 0)
      OriginalInstructionAddresses.erase(PC);
    I->eraseFromParent();
  }

  /// \brief Drop \p Start and all the descendants, stopping when a JT is met
  void purgeTranslation(llvm::BasicBlock *Start);

  /// \brief Check if \p BB has at least a predecessor, excluding the dispatcher
  bool hasPredecessors(llvm::BasicBlock *BB) const;

  /// \brief Rebuild the dispatcher switch
  ///
  /// Depending on the CFG form we're currently adopting the dispatcher might go
  /// to all the jump targets or only to those who have no other predecessor.
  void rebuildDispatcher();

  // TODO: instead of a gigantic switch case we could map the original memory
  //       area and write the address of the translated basic block at the jump
  //       target
  void
  createDispatcher(llvm::Function *OutputFunction, llvm::Value *SwitchOnPtr);

  template<typename value_type, unsigned endian>
  void findCodePointers(uint64_t StartVirtualAddress,
                        const unsigned char *Start,
                        const unsigned char *End);

  void harvest();

  void handleSumJump(llvm::Instruction *SumJump);

private:
  using BlockMap = std::map<uint64_t, JumpTarget>;
  using InstructionMap = std::map<uint64_t, llvm::Instruction *>;

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
  llvm::Value *PCReg;
  llvm::Function *ExitTB;
  RangesVector ExecutableRanges;
  llvm::BasicBlock *Dispatcher;
  llvm::SwitchInst *DispatcherSwitch;
  llvm::BasicBlock *DispatcherFail;
  llvm::BasicBlock *AnyPC;
  llvm::BasicBlock *UnexpectedPC;
  std::set<llvm::BasicBlock *> Visited;

  const BinaryFile &Binary;

  unsigned NewBranches = 0;

  std::set<uint64_t> UnusedCodePointers;
  interval_set ReadIntervalSet;
  NoReturnAnalysis NoReturn;

  CFGForm::Values CurrentCFGForm;
  std::set<llvm::BasicBlock *> ToPurge;
  std::set<uint64_t> SimpleLiterals;
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

#endif // JUMPTARGETMANAGER_H
