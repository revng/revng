//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"

#include "revng/ADT/SmallMap.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

using llvm::AllocaInst;
using llvm::AnalysisUsage;
using llvm::BasicBlock;
using llvm::Function;
using llvm::FunctionPass;
using llvm::IRBuilder;
using llvm::PHINode;
using llvm::RegisterPass;
using llvm::Use;
using llvm::User;
using llvm::Value;

static Logger<> Log{ "exit-ssa" };

struct ExitSSAPass : public FunctionPass {
public:
  static char ID;

  ExitSSAPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }
};

static std::vector<std::set<PHINode *>> getPHIEquivalenceClasses(Function &F) {

  llvm::EquivalenceClasses<PHINode *> PHIEquivalenceClasses;

  for (BasicBlock &BB : F) {
    for (auto &PHI : BB.phis()) {

      PHIEquivalenceClasses.insert(&PHI);

      for (User *U : PHI.users()) {
        if (auto *PHIUser = dyn_cast<PHINode>(U)) {
          PHIEquivalenceClasses.insert(PHIUser);
          PHIEquivalenceClasses.unionSets(&PHI, PHIUser);
        }
      }

      for (Value *Incoming : PHI.incoming_values()) {
        if (auto *PHIIncoming = dyn_cast<PHINode>(Incoming)) {
          PHIEquivalenceClasses.insert(PHIIncoming);
          PHIEquivalenceClasses.unionSets(&PHI, PHIIncoming);
        }
      }
    }
  }

  std::vector<std::set<PHINode *>> Result;

  auto I = PHIEquivalenceClasses.begin();
  auto E = PHIEquivalenceClasses.end();
  // Iterate over all of the members.
  for (; I != E; ++I) {

    // Ignore all the members that are not leaders of a class.
    if (not I->isLeader())
      continue;

    // The interate all over the elements of a class, and build the set of
    // PHINodes that represent that class.
    std::set<PHINode *> PHISet;
    auto PHIRange = llvm::make_range(PHIEquivalenceClasses.member_begin(I),
                                     PHIEquivalenceClasses.member_end());
    for (PHINode *PHI : PHIRange)
      PHISet.insert(PHI);

    Result.push_back(std::move(PHISet));
  }

  return Result;
}

static void
replacePHIEquivalenceClass(const std::set<PHINode *> &PHIs, Function &F) {

  revng_log(Log, "New PHIGroup ================");
  LoggerIndent FirstIndent{ Log };

  IRBuilder<> Builder(F.getContext());
  Builder.SetInsertPointPastAllocas(&F);
  AllocaInst *Alloca = Builder.CreateAlloca((*PHIs.begin())->getType());
  revng_log(Log, "Created Alloca: " << dumpToString(Alloca));

  {
    revng_log(Log, "Replacing Incomings");
    LoggerIndent IndentIncomings{ Log };

    // First, we replace all the incoming that are not PHIs with stores of the
    // incoming in the associated local variable.
    for (auto *PHI : PHIs) {

      revng_log(Log, "PHI: " << dumpToString(PHI));
      LoggerIndent IndentPHI{ Log };

      for (Use &IncomingUse : PHI->incoming_values()) {
        Value *Incoming = IncomingUse.get();
        revng_log(Log, "Incoming: " << dumpToString(Incoming));

        // Skip PHINodes. They are already part of PHIs and they will be
        // handled in consequent operations.
        if (llvm::isa<PHINode>(Incoming))
          continue;

        llvm::BasicBlock *BB = PHI->getIncomingBlock(IncomingUse);
        Builder.SetInsertPoint(BB->getTerminator());
        auto *S = Builder.CreateStore(Incoming, Alloca);
        revng_log(Log, dumpToString(S));
      }
    }
  }

  {
    revng_log(Log, "Replacing Uses");
    LoggerIndent IndentUses{ Log };

    // Then, for all Uses whose Users are not also PHINodes we replace them with
    // a load.
    // All the uses that are PHIs are replaced with undef instead, and they will
    // be cleaned up later.
    for (auto *PHI : PHIs) {

      revng_log(Log, "Use of PHI: " << dumpToString(PHI));
      LoggerIndent IndentPHI{ Log };

      Builder.SetInsertPoint(PHI);
      auto *NewLoad = createLoad(Builder, Alloca);
      revng_log(Log, "Create new load: " << dumpToString(NewLoad));

      for (Use &U : llvm::make_early_inc_range(PHI->uses())) {
        revng_log(Log, "in User: " << dumpToString(U.getUser()));

        Value *NewOperand = nullptr;
        if (isa<PHINode>(U.getUser()))
          NewOperand = llvm::UndefValue::get(PHI->getType());
        else
          NewOperand = NewLoad;
        revng_log(Log, "replaced with: " << dumpToString(NewOperand));
        U.set(NewOperand);
      }

      if (not NewLoad->getNumUses()) {
        revng_log(Log, "Erase new load since it has 0 uses");
        NewLoad->eraseFromParent();
      }
    }
  }

  // Finally we remove all the PHIs
  {
    revng_log(Log, "Cleaning Up");
    LoggerIndent IndentCleanup{ Log };
    for (auto *PHI : PHIs) {
      revng_log(Log, "Erasing: " << dumpToString(PHI));
      PHI->eraseFromParent();
    }
  }
}

bool ExitSSAPass::runOnFunction(Function &F) {

  // Skip non-isolated functions
  if (not FunctionTags::Isolated.isTagOf(&F))
    return false;

  bool Changed = false;

  // A vector containing sets of equivalence classes of PHINodes.
  // Each equivalence class is composed of connected PHINodes that can form
  // trees, a DAGs, or even loops.
  // Informally, all the PHINodes in a group hold the same value, and we want to
  // create a single local variable for each DAG.
  const auto PHIClasses = getPHIEquivalenceClasses(F);
  for (const auto &PHIGroup : PHIClasses)
    replacePHIEquivalenceClass(PHIGroup, F);
  Changed |= not PHIClasses.empty();

  return Changed;
}

char ExitSSAPass::ID = 0;

static RegisterPass<ExitSSAPass> X("exit-ssa",
                                   "Transformation pass that exits from Static "
                                   "Single Assignment form, promoting PHINodes "
                                   "to sets of Allocas, Load and Stores",
                                   false,
                                   false);
