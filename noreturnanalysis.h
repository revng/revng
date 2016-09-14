#ifndef _NORETURNANALYSIS_H
#define _NORETURNANALYSIS_H

// Standard includes
#include <cstdint>
#include <map>
#include <vector>

// LLVM includes
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/StringRef.h"

// Local includes
#include "reachingdefinitions.h"
#include "revamb.h"

namespace llvm {
class BasicBlock;
class CallInst;
class Instruction;
class LoadInst;
class StoreInst;
}

class NoReturnAnalysis {
public:
  NoReturnAnalysis(Architecture TheArchitecture)
    : SourceArchitecture(TheArchitecture), NoDCE(nullptr) { }

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
  void computeKillerSet(PredecessorsMap &CallPredecessors,
                        std::set<llvm::TerminatorInst *> &Returns);

  void setDispatcher(llvm::BasicBlock *BB) { Dispatcher = BB; }

  /// \brief Check if the given basic block as has been registered as a killer
  bool isNoreturnBasicBlock(llvm::BasicBlock *BB) {
    for (llvm::BasicBlock *Successor : successors(BB))
      if (!isKiller(Successor))
        return false;
    return true;
  }

private:
  bool isKiller(llvm::BasicBlock *BB) const {
    return KillerBBs.count(BB) != 0;
  };

  bool endsUpIn(llvm::Instruction *I, llvm::BasicBlock *Target);

  bool checkKiller(llvm::BasicBlock *BB) const {
    if (BB == Dispatcher)
      return false;

    for (llvm::BasicBlock *Successor : successors(BB))
      if (BB != Dispatcher && KillerBBs.count(Successor) == 0)
        return false;
    return true;
  }

private:
  Architecture SourceArchitecture;
  std::set<llvm::CallInst *> RegisteredSyscalls;
  std::vector<llvm::LoadInst *> SyscallRegisterReads;
  std::set<llvm::StoreInst *> SyscallRegisterDefinitions;
  std::set<llvm::BasicBlock *> KillerBBs;
  llvm::BasicBlock *Dispatcher;
  llvm::Function *NoDCE;
};

#endif // _NORETURNANALYSIS_H
