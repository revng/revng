#ifndef NORETURNANALYSIS_H
#define NORETURNANALYSIS_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdint>
#include <map>
#include <vector>

// LLVM includes
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"

// Local libraries includes
#include "revng/ReachingDefinitions/ReachingDefinitionsPass.h"
#include "revng/Support/revng.h"

namespace llvm {
class BasicBlock;
class CallInst;
class Instruction;
class LoadInst;
class StoreInst;
class TerminatorInst;
} // namespace llvm

class NoReturnAnalysis {
public:
  NoReturnAnalysis(Architecture TheArchitecture) :
    SourceArchitecture(TheArchitecture),
    NoDCE(nullptr) {}

  /// Records all the calls to the syscall helper and inject a sentinel load
  /// from the syscall number register
  void registerSyscalls(llvm::Function *F);

  /// \brief Use \p CRL to collect all the definitions reaching the sentinel
  ///        load
  void collectDefinitions(ConditionalReachedLoadsPass &CRL);

  /// \brief Return true if \p Store is ever used to write the syscall number
  bool setsSyscallNumber(llvm::StoreInst *Store);

  /// \brief If appropriate, register \p Setter's basic block as a killer
  ///
  /// \param StoredValue the value stored by \p Definition starting from \p
  ///        Setter
  /// \param Setter the instruction responsible for making \p Definition store
  ///        \p StoredValue
  /// \param Definition the definition that will end up in the syscall register
  // TODO: check from Setter you have to get to Definition
  void registerKiller(uint64_t StoredValue,
                      llvm::Instruction *Setter,
                      llvm::Instruction *Definition);

  using PredecessorsMap = std::map<llvm::BasicBlock *,
                                   std::vector<llvm::BasicBlock *>>;
  /// Add to the set of killer basic blocks all the basic blocks who can only
  /// end in one of those already registered.
  void computeKillerSet(PredecessorsMap &CallPredecessors);

  void setDispatcher(llvm::BasicBlock *BB) { Dispatcher = BB; }

  /// \brief Check if the given basic block as has been registered as a killer
  bool isNoreturnBasicBlock(llvm::BasicBlock *BB) {
    for (llvm::BasicBlock *Successor : successors(BB))
      if (!isKiller(Successor))
        return false;
    return true;
  }

  void cleanup() {
    // Cleanup all the calls to "nodce"
    if (NoDCE != nullptr) {
      for (llvm::User *NoDCEUser : NoDCE->users())
        llvm::cast<llvm::CallInst>(NoDCEUser)->eraseFromParent();

      NoDCE->eraseFromParent();
    }
  }

private:
  bool isKiller(llvm::BasicBlock *BB) const {
    return KillerBBs.count(BB) != 0;
  };

  /// \brief Register BB as killer and associate a noreturn metadata to it
  void registerKiller(llvm::BasicBlock *BB, KillReason::Values Reason) {
    KillerBBs.insert(BB);

    if (!BB->empty()) {
      llvm::TerminatorInst *Terminator = BB->getTerminator();
      revng_assert(Terminator != nullptr);
      if (Terminator->getMetadata("noreturn") == nullptr) {
        QuickMetadata QMD(getContext(BB));
        Terminator->setMetadata("noreturn",
                                QMD.tuple(KillReason::getName(Reason)));
      }
    }
  }

  bool endsUpIn(llvm::Instruction *I, llvm::BasicBlock *Target);

  bool checkKiller(llvm::BasicBlock *BB) const {
    if (BB == Dispatcher)
      return false;

    for (llvm::BasicBlock *Successor : successors(BB))
      if (BB != Dispatcher && KillerBBs.count(Successor) == 0)
        return false;
    return true;
  }

  bool hasSyscalls() const { return NoDCE != nullptr; }

  /// \brief Register as killer basic blocks those parts of infinite loops
  void findInfinteLoops();

private:
  Architecture SourceArchitecture;
  std::set<llvm::CallInst *> RegisteredSyscalls;
  std::vector<llvm::LoadInst *> SyscallRegisterReads;
  std::set<llvm::StoreInst *> SyscallRegisterDefinitions;
  std::set<llvm::BasicBlock *> KillerBBs;
  llvm::BasicBlock *Dispatcher;
  llvm::Function *NoDCE;
};

#endif // NORETURNANALYSIS_H
