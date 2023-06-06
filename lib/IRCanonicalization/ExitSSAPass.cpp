//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <compare>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"

#include "revng/ADT/SmallMap.h"
#include "revng/ADT/ZipMapIterator.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

#include "MarkAssignments/LivenessAnalysis.h"

using namespace llvm;

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

static std::vector<std::set<PHINode *>>
getPHIEquivalenceClasses(Function &F,
                         const LivenessAnalysis::LivenessMap &LiveIn) {

  // PHINodes in the same class are mapped onto the same local variable.
  llvm::EquivalenceClasses<PHINode *> PHISameVariableClasses;

  std::unordered_map<const PHINode *, std::set<const BasicBlock *>>
    VariableLiveSet;

  auto RPOT = llvm::ReversePostOrderTraversal(&F);

  for (const BasicBlock *BB : RPOT) {

    for (const PHINode &PHI : BB->phis())
      VariableLiveSet[&PHI] = {};

    auto LiveIt = LiveIn.find(BB);
    if (LiveIt == LiveIn.end())
      continue;

    for (const Instruction *I : LiveIn.at(BB))
      if (const PHINode *PHI = dyn_cast<const PHINode>(I))
        VariableLiveSet[PHI].insert(BB);
  }

  const auto InitVariableClass = [&PHISameVariableClasses](PHINode *PHI) {
    if (PHISameVariableClasses.findValue(PHI) != PHISameVariableClasses.end())
      return;

    PHISameVariableClasses.insert(PHI);
    return;
  };

  for (BasicBlock *BB : RPOT) {
    for (auto &PHI : BB->phis()) {

      // Set up an equivalence class for PHI, if necessary
      InitVariableClass(&PHI);

      // Then, for each user, if it's a PHINode, try to see if we can insert it
      // in the same equivalence class as PHI.
      for (User *U : PHI.users()) {
        auto *PHIUser = dyn_cast<PHINode>(U);
        if (not PHIUser or PHIUser == &PHI)
          continue;

        // Set up an equivalence class for PHIUser, if necessary.
        // Sometimes this might not be necessary, because we might have already
        // seen the PHIUser in case of loops. If this happens everything is
        // already set up for the PHIUser and the following call is a nop. But
        // we still have to do it because otherwise the following isEquivalent
        // call might fail.
        InitVariableClass(PHIUser);

        // If PHI and PHIUser are already in the same equivalence class, there's
        // nothing to do.
        if (PHISameVariableClasses.isEquivalent(&PHI, PHIUser))
          continue;

        PHINode *PHILeader = PHISameVariableClasses.getLeaderValue(&PHI);
        PHINode *UserLeader = PHISameVariableClasses.getLeaderValue(PHIUser);

        // Now let's see if there are conflicting live sets.
        auto PHILiveSetIt = VariableLiveSet.find(PHILeader);
        revng_assert(PHILiveSetIt != VariableLiveSet.end());
        auto UserLiveSetIt = VariableLiveSet.find(UserLeader);
        revng_assert(UserLiveSetIt != VariableLiveSet.end());

        // If there are conflicting live sets it means that the two sets of PHIs
        // hold different values that must be kept alive at the same time,
        // otherwise we'll lose one of them on at least one path.
        // In this case we have to bail out.
        if (not llvm::set_intersection(PHILiveSetIt->second,
                                       UserLiveSetIt->second)
                  .empty())
          continue;

        // Here the two are compatible so we join the equivalence classes.
        PHISameVariableClasses.unionSets(&PHI, PHIUser);

        // Finally we do the same with the live ranges.
        {
          auto Handle = VariableLiveSet.extract(UserLiveSetIt);
          PHILiveSetIt->second.merge(std::move(Handle.mapped()));
        }
      }
    }
  }

  std::vector<std::set<PHINode *>> Result;

  auto I = PHISameVariableClasses.begin();
  auto E = PHISameVariableClasses.end();
  // Iterate over all of the members.
  for (; I != E; ++I) {

    // Ignore all the members that are not leaders of a class.
    if (not I->isLeader())
      continue;

    // Then iterate all over the elements of a class, and build the set of
    // PHINodes that represent that class.
    std::set<PHINode *> PHISet;
    auto PHIRange = llvm::make_range(PHISameVariableClasses.member_begin(I),
                                     PHISameVariableClasses.member_end());
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

      // TODO: if there are duplicated incoming values from different blocks, we
      // should think of inserting a new BasicBlock with only one single store
      // instead of many StoreInst, one for each incoming block.
      for (Use &IncomingUse : PHI->incoming_values()) {
        Value *Incoming = IncomingUse.get();
        revng_log(Log, "Incoming: " << dumpToString(Incoming));

        // Skip PHINodes. They are already part of PHIs and they will be
        // handled in consequent operations.
        if (auto *IncomingPHI = dyn_cast<PHINode>(Incoming);
            Incoming and PHIs.contains(IncomingPHI))
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
          NewOperand = UndefValue::get(PHI->getType());
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

  LivenessAnalysis::LivenessMap LiveIn = computeLiveness(F);

  // A vector containing sets of equivalence classes of PHINodes.
  // Each equivalence class is composed of connected PHINodes that can form
  // trees, a DAGs, or even loops.
  // Informally, all the PHINodes in a group hold the same value, and we want to
  // create a single local variable for each DAG.
  const auto PHIClasses = getPHIEquivalenceClasses(F, LiveIn);
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
