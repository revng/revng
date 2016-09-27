/// \file noreturnanalysis.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

// Local includes
#include "datastructures.h"
#include "debug.h"
#include "ir-helpers.h"
#include "noreturnanalysis.h"

using namespace llvm;

void NoReturnAnalysis::registerSyscalls(llvm::Function *F) {
  // Look for calls to the syscall helper
  Module *M = F->getParent();
  Function *SyscallHandler = M->getFunction(SourceArchitecture.syscallHelper());
  if (SyscallHandler == nullptr)
    return;

  StringRef RegisterName = SourceArchitecture.syscallNumberRegister();
  Value *SyscallNumberRegister = M->getGlobalVariable(RegisterName);
  if (SyscallNumberRegister == nullptr)
    return;

  // Lazily create the "nodce" function
  if (NoDCE == nullptr) {
    Type *VoidTy = Type::getVoidTy(M->getContext());
    Type *SNRTy = SyscallNumberRegister->getType()->getPointerElementType();
    NoDCE = cast<Function>(M->getOrInsertFunction("nodce",
                                                  VoidTy,
                                                  SNRTy,
                                                  nullptr));
  }

  for (User *U : SyscallHandler->users()) {
    if (auto *Call = dyn_cast<CallInst>(U)) {
      // Consider calls only in the requested function
      if (Call->getParent()->getParent() != F)
        continue;

      // If not done already, register the syscall and inject a dead load from
      // the register of the syscall number. The load is also used as a
      // parameter to a "nodce" function to prevent the DCE from removing it.
      if (RegisteredSyscalls.count(Call) == 0) {
        auto *DeadLoad = new LoadInst(SyscallNumberRegister, "", Call);
        CallInst::Create(NoDCE, { DeadLoad }, "", Call);
        SyscallRegisterReads.push_back(DeadLoad);
        RegisteredSyscalls.insert(Call);
      }

    }
  }

}

bool NoReturnAnalysis::endsUpIn(Instruction *I, BasicBlock *Target) {
  // Very simple check if the only possible destination of I is Target.
  // The only check we make is if there's always a single sucessor which in the
  // end leads to Target.
  // TODO: once we enforce the CFG we should not ignore the dispatcher
  // TODO: expand
  BasicBlock *BB = I->getParent();
  std::set<BasicBlock *> Visited;
  while (BB != Target) {
    if (BB == nullptr || Visited.count(BB) != 0)
      return false;
    Visited.insert(BB);

    BasicBlock *Next = nullptr;
    for (BasicBlock *Successor : successors(BB)) {
      if (Successor != Dispatcher) {
        if (Next == nullptr) {
          Next = Successor;
        } else {
          return false;
        }
      }
    }

    BB = Next;
  }

  return true;
}

void NoReturnAnalysis::collectDefinitions(ConditionalReachedLoadsPass &CRL) {
  if (!hasSyscalls())
    return;

  // Cleanup all the calls to "nodce"
  for (User *NoDCEUser : NoDCE->users())
    cast<CallInst>(NoDCEUser)->eraseFromParent();

  SyscallRegisterDefinitions.clear();
  KillerBBs.clear();

  // Register all the reaching definitions of the SyscallRegisterReads, which
  // can only end up in the corresponding read
  for (LoadInst *Load : SyscallRegisterReads)
    for (Instruction *I : CRL.getReachingDefinitions(Load))
      if (auto *Store = dyn_cast<StoreInst>(I))
        if (endsUpIn(Store, Load->getParent()))
            SyscallRegisterDefinitions.insert(Store);
}

bool NoReturnAnalysis::setsSyscallNumber(llvm::StoreInst *Store) {
  return SyscallRegisterDefinitions.count(Store) != 0;
}

void NoReturnAnalysis::registerKiller(uint64_t StoredValue,
                                      Instruction *Setter,
                                      Instruction *Definition) {
  // Consider this killer only if setter can only go into definition
  // TODO: is the necessary?
  if (!endsUpIn(Setter, Definition->getParent()))
    return;

  // Check if StoredValue is a noreturn syscall
  auto NRS = SourceArchitecture.noReturnSyscalls();
  auto NRSIt = std::find(std::begin(NRS), std::end(NRS), StoredValue);
  if (NRSIt == std::end(NRS))
    return;

  // Register the Setter's basic block
  KillerBBs.insert(Setter->getParent());
}

void NoReturnAnalysis::computeKillerSet(PredecessorsMap &CallPredecessors,
                                        std::set<TerminatorInst *> &Returns) {
  // Visit every predecessor once
  OnceQueue<BasicBlock *> WorkList;
  for (BasicBlock *KillerBB : KillerBBs)
    WorkList.insert(KillerBB);

  while (!WorkList.empty()) {
    BasicBlock *BB = WorkList.pop();

    // Skip function calls
    auto CallPredecessorIt = CallPredecessors.find(BB);
    if (CallPredecessorIt != CallPredecessors.end()) {

      for (BasicBlock *Pred : CallPredecessorIt->second) {
        KillerBBs.insert(Pred);
        WorkList.insert(Pred);
      }

    }

    // Check all the predecessors unless they're return basic blocks
    for (BasicBlock *Pred : predecessors(BB)) {
      if (Returns.count(BB->getTerminator()) == 0 && checkKiller(Pred)) {
        KillerBBs.insert(Pred);
        WorkList.insert(Pred);
      }
    }

  }

  DBG("nra", {
      for (BasicBlock *KillerBB : KillerBBs)
        dbg << getName(KillerBB) << " is killer BB\n";
    });

}
