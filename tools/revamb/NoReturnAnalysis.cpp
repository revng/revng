/// \file noreturnanalysis.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

// Local libraries includes
#include "revng/ADT/Queue.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

// Local includes
#include "NoReturnAnalysis.h"

using namespace llvm;

Logger<> NRALog("nra");

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
    auto *FunctionC = M->getOrInsertFunction("nodce", VoidTy, SNRTy);
    NoDCE = cast<Function>(FunctionC);
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
  registerKiller(Setter->getParent(), KillReason::KillerSyscall);
}

void NoReturnAnalysis::findInfinteLoops() {
  Function *F = Dispatcher->getParent();
  DominatorTreeBase<BasicBlock, /* IsPostDom = */ false> DT;
  DT.recalculate(*F);

  LoopInfo LI(DT);
  for (Loop *L : LI) {
    SmallVector<BasicBlock *, 3> ExitingBlocks;
    L->getExitingBlocks(ExitingBlocks);
    if (ExitingBlocks.size() == 0) {
      for (BasicBlock *Member : L->blocks())
        registerKiller(Member, KillReason::EndlessLoop);
    }
  }
}

void NoReturnAnalysis::computeKillerSet(PredecessorsMap &CallPredecessors) {
  // Enrich the KillerBBs set with blocks participating in infinite loops
  findInfinteLoops();

  if (KillerBBs.size() == 0)
    return;

  // Make all the killer basic blocks jump to a "sink" basic block, so that
  // we can easily identify all the other killer basic blocks using the post
  // dominator tree.
  //
  // Note: computing the post-dominated basic blocks from the sink is different
  // from computing the union of all the basic blocks post-dominated by a killer
  // basic block.
  Function *F = Dispatcher->getParent();
  LLVMContext &C = F->getParent()->getContext();
  auto *Sink = BasicBlock::Create(C, "sink", F);
  new UnreachableInst(C, Sink);

  std::vector<std::pair<BasicBlock *, TerminatorInst *>> Backup;
  for (BasicBlock *KillerBB : KillerBBs) {
    // Save a reference to the original terminator and detach it from the basic
    // block
    TerminatorInst *Terminator = KillerBB->getTerminator();
    Backup.push_back({ KillerBB, Terminator });
    Terminator->removeFromParent();

    // Replace the old terminator with an unconditional branch to the sink
    // TODO: this is dangerous, we might break some relevant edge
    BranchInst::Create(Sink, KillerBB);
  }

  // Compute the post-dominator tree on the CFG (in NoFunctionCallsCFG state)
  DominatorTreeBase<BasicBlock, /* IsPostDom = */ true> PDT;
  PDT.recalculate(*F);

  // The worklist initially contains only the sink but will be populated with
  // the basic blocks calling a killer basic block (i.e., function)
  OnceQueue<BasicBlock *> WorkList;
  WorkList.insert(Sink);

  while (!WorkList.empty()) {
    BasicBlock *BB = WorkList.pop();

    // Collect all the post-dominated basic blocks
    SmallVector<BasicBlock *, 5> Descendants;
    PDT.getDescendants(BB, Descendants);

    // These are all killer basic blocks
    for (BasicBlock *NewKiller : Descendants)
      registerKiller(NewKiller, KillReason::LeadsToKiller);

    // Find all the callers, mark them as killers and enqueue them for
    // processing since all the basic block dominated by the callee basic block
    // are killer basic blocks too
    for (BasicBlock *NewKiller : Descendants) {
      for (User *U : NewKiller->users()) {
        auto *Address = dyn_cast<BlockAddress>(U);
        if (Address == nullptr)
          continue;

        for (User *U : Address->users()) {
          auto *Call = dyn_cast<CallInst>(U);
          if (Call == nullptr)
            continue;

          Function *Callee = Call->getCalledFunction();
          if (Callee == nullptr || Callee->getName() != "function_call")
            continue;

          BasicBlock *CallerBB = Call->getParent();
          if (!isKiller(CallerBB)) {
            // TODO: support multiple successors, i.e. check they're all killers
            registerKiller(CallerBB, KillReason::LeadsToKiller);
            WorkList.insert(CallerBB);
          }
        }
      }
    }
  }

  // Restore the backup
  for (auto &P : Backup) {
    BasicBlock *KillerBB = P.first;
    TerminatorInst *Terminator = P.second;
    KillerBB->getTerminator()->eraseFromParent();
    if (KillerBB->empty()) {
      auto &List = KillerBB->getInstList();
      List.insert(List.begin(), Terminator);
    } else {
      Terminator->insertAfter(&KillerBB->back());
    }
  }

  // We no longer need the sink
  Sink->eraseFromParent();

  if (NRALog.isEnabled())
    for (BasicBlock *KillerBB : KillerBBs)
      NRALog << getName(KillerBB) << " is killer BB" << DoLog;
}
