//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>
#include <compare>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"

#include "revng/ADT/SmallMap.h"
#include "revng/ADT/ZipMapIterator.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRBuilder.h"
#include "revng/Support/IRHelpers.h"

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

struct IncomingInfo {
  BasicBlock *PHIBlock = nullptr;
  BasicBlock *IncomingBlock = nullptr;
  Value *IncomingValue = nullptr;
  bool operator==(const IncomingInfo &) const = default;
  std::strong_ordering operator<=>(const IncomingInfo &) const = default;
};

static bool haveIncompatibleIncomings(const std::set<IncomingInfo> &LHS,
                                      const std::set<IncomingInfo> &RHS) {
  for (const auto &[PHIBlock, IncomingBlock, IncomingValue] : LHS) {
    auto It = RHS.lower_bound(IncomingInfo{ PHIBlock, IncomingBlock, nullptr });
    auto End = RHS.upper_bound(IncomingInfo{
      PHIBlock, IncomingBlock, std::numeric_limits<Value *>::max() });
    // If RHS contains a PHI that is in the same block as PHIBlock, and has a
    // different incoming value on the same incoming block, the two are
    // incompatible, because they would assign two different values to the same
    // local variable along the same edge.
    if (It != End and IncomingValue != It->IncomingValue)
      return false;
  }
  return true;
}

static std::vector<SetVector<PHINode *>> getPHIEquivalenceClasses(Function &F) {

  // PHINodes in the same class are mapped onto the same local variable.
  llvm::EquivalenceClasses<PHINode *> PHISameVariableClasses;

  std::unordered_map<PHINode *, std::set<IncomingInfo>> PerClassIncomings;

  const auto InitVariableClass = [&PHISameVariableClasses,
                                  &PerClassIncomings](PHINode *PHI) {
    if (PHISameVariableClasses.findValue(PHI) != PHISameVariableClasses.end())
      return;

    PHISameVariableClasses.insert(PHI);

    auto &CurrentIncomingInfo = PerClassIncomings[PHI];
    unsigned NumIncomings = PHI->getNumIncomingValues();
    BasicBlock *PHIBlock = PHI->getParent();
    for (unsigned I = 0U; I < NumIncomings; ++I) {
      Value *IncomingValue = PHI->getIncomingValue(I);
      BasicBlock *IncomingBlock = PHI->getIncomingBlock(I);
      auto NewIncomingInfo = IncomingInfo{ PHIBlock,
                                           IncomingBlock,
                                           IncomingValue };
      CurrentIncomingInfo.insert(std::move(NewIncomingInfo));
    }
    return;
  };

  for (BasicBlock *BB : llvm::ReversePostOrderTraversal(&F)) {
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

        // If the PHI has a user that is not another PHI, it cannot be put in
        // the same equivalence class as the PHIUser, so we bail out.
        if (llvm::any_of(PHI.users(),
                         [](const User *U) { return not isa<PHINode>(U); }))
          continue;

        PHINode *PHILeader = PHISameVariableClasses.getLeaderValue(&PHI);
        PHINode *UserLeader = PHISameVariableClasses.getLeaderValue(PHIUser);

        // Now let's see if there are conflicting live sets.
        auto PHIIncomingInfo = PerClassIncomings.find(PHILeader);
        revng_assert(PHIIncomingInfo != PerClassIncomings.end());
        auto UserIncomingInfo = PerClassIncomings.find(UserLeader);
        revng_assert(UserIncomingInfo != PerClassIncomings.end());

        // If there are conflicting incoming it means that the two sets of PHIs
        // hold different values that must be kept alive at the same time,
        // otherwise we'll lose one of them. In this case we have to bail out.
        if (haveIncompatibleIncomings(PHIIncomingInfo->second,
                                      UserIncomingInfo->second))
          continue;

        // Here the two are compatible so we join the equivalence classes.
        PHISameVariableClasses.unionSets(&PHI, PHIUser);

        // Finally we do the same with the IncomingInfo
        auto Handle = PerClassIncomings.extract(UserIncomingInfo);
        PHIIncomingInfo->second.merge(std::move(Handle.mapped()));
      }
    }
  }

  std::vector<SetVector<PHINode *>> Result;

  // We want to return the equivalence classes in deterministic order.
  // Sort them according to the RPOT order of their leader.
  auto ClassEnd = PHISameVariableClasses.end();
  for (BasicBlock *BB : llvm::post_order(&F)) {
    for (PHINode &PHI : BB->phis()) {
      auto ClassIterator = PHISameVariableClasses.findValue(&PHI);
      revng_assert(ClassIterator != ClassEnd);
      if (not ClassIterator->isLeader())
        continue;

      // If we found a leader, iterate all over the elements of a class, and
      // build the set of PHINodes that represent that class.
      auto PHIRange = llvm::make_range(PHISameVariableClasses
                                         .member_begin(ClassIterator),
                                       PHISameVariableClasses.member_end());

      // Things are pushed into Result in deterministic order because we're
      // iterating in post_order over the Function and considering only leader
      // PHINodes, in program order.
      Result.push_back({});

      // The iteration on the PHIRange is deterministic, because it depends only
      // on the order how elements were inserted in the class, and that in turns
      // is deterministic because we do it in a deterministic order in RPO above
      SetVector<PHINode *> &PHIs = Result.back();
      for (PHINode *PHI : PHIRange)
        PHIs.insert(PHI);
    }
  }

  return Result;
}

static bool isIncomingValueUseInPHI(const Use &U) {
  auto *PHIUser = dyn_cast<PHINode>(U.getUser());
  if (nullptr == PHIUser)
    return false;

  unsigned OpNumber = U.getOperandNo();
  if (OpNumber >= PHIUser->getNumOperands())
    return false;

  unsigned IncomingNumber = PHINode::getIncomingValueNumForOperand(OpNumber);
  Value *IncomingVal = PHIUser->getIncomingValue(IncomingNumber);

  return IncomingVal == U.get();
}

using UseSet = llvm::SmallSet<Use *, 2>;

static auto
getIncomingUsesOfValuesFromBlocks(const SetVector<PHINode *> &PHIs) {
  llvm::MapVector<std::pair<BasicBlock *, Value *>, UseSet>
    IncomingUsesOfValueFromBlock;

  for (auto *PHI : PHIs) {
    for (Use &IncomingUse : PHI->incoming_values()) {
      Value *Incoming = IncomingUse;
      if (isa<llvm::UndefValue>(Incoming))
        continue;
      // If the incoming is internal to the equivalence class (PHIs) we ignore
      // it.
      if (auto *PHIIncoming = dyn_cast<PHINode>(Incoming);
          PHIIncoming and PHIs.contains(PHIIncoming)
          and isIncomingValueUseInPHI(IncomingUse))
        continue;

      BasicBlock *IncomingBlock = PHI->getIncomingBlock(IncomingUse);
      IncomingUsesOfValueFromBlock[std::make_pair(IncomingBlock, Incoming)]
        .insert(&IncomingUse);
    }
  }

  // Then we sort everything so that entries with the same BasicBlock are
  // contiguous, and the first Value in a given block is the one with the
  // highest number of uses.
  auto Result = IncomingUsesOfValueFromBlock.takeVector();

  const auto Cmp =
    [](const std::pair<std::pair<BasicBlock *, Value *>, UseSet> &LHS,
       const std::pair<std::pair<BasicBlock *, Value *>, UseSet> &RHS) {
      const auto &[LHSBlockAndValue, LHSUses] = LHS;
      const auto &[LHSBlock, LHSValue] = LHSBlockAndValue;
      const auto &[RHSBlockAndValue, RHSUses] = RHS;
      const auto &[RHSBlock, RHSValue] = RHSBlockAndValue;

      // Ordering of blocks is not important per se, but it's important that
      // entries with the same block are sorted in a contiguous range.
      if (auto CmpBlocks = LHSBlock <=> RHSBlock; CmpBlocks != 0)
        return CmpBlocks < 0;

      // Soft first the entry with the largest number of uses.
      return LHSUses.size() > RHSUses.size();
    };
  llvm::stable_sort(Result, Cmp);
  return Result;
}

using EdgeToNewBlockMap = std::map<std::pair<BasicBlock *, BasicBlock *>,
                                   BasicBlock *>;

static void
buildStore(BasicBlock *StoreBlock, Value *Incoming, AllocaInst *Alloca) {
  // TODO: the checks should be enabled conditionally based on the user.
  revng::NonDebugInfoCheckingIRBuilder Builder(StoreBlock->getContext());

  auto *IncomingInst = dyn_cast<Instruction>(Incoming);
  if (IncomingInst and IncomingInst->getParent() == StoreBlock) {
    BasicBlock *IncomingParentBlock = IncomingInst->getParent();
    if (isa<AllocaInst>(IncomingInst)) {
      Function *ParentFunction = StoreBlock->getParent();
      revng_assert(IncomingParentBlock == &ParentFunction->getEntryBlock());
      Builder.SetInsertPointPastAllocas(ParentFunction, Alloca->getDebugLoc());
    } else {
      Builder.SetInsertPoint(StoreBlock,
                             std::next(IncomingInst->getIterator()),
                             Alloca->getDebugLoc());
    }
  } else {
    Builder.SetInsertPoint(StoreBlock->getTerminator(), Alloca->getDebugLoc());
  }

  auto *S = Builder.CreateStore(Incoming, Alloca);
  revng_log(Log,
            "Created StoreInst " << dumpToString(S)
                                 << " in Block: " << StoreBlock->getName());

  if (IncomingInst and IncomingInst->getParent() == StoreBlock) {
    Instruction *LoadFromStore = nullptr;
    for (Instruction &NextInBlock :
         llvm::make_range(std::next(S->getIterator()), StoreBlock->end())) {
      for (Use &Operand : NextInBlock.operands()) {
        if (Operand.get() == IncomingInst) {
          if (not LoadFromStore) {
            LoadFromStore = Builder.CreateLoad(IncomingInst->getType(), Alloca);
            if (auto *IncomingInst = dyn_cast<Instruction>(Incoming))
              LoadFromStore->setDebugLoc(IncomingInst->getDebugLoc());
          }
          Operand.set(LoadFromStore);
        }
      }
    }
  }
}

static void replacePHIEquivalenceClass(const SetVector<PHINode *> &PHIs,
                                       Function &F,
                                       EdgeToNewBlockMap &NewBlocks) {

  revng_log(Log, "New PHIGroup ================");
  LoggerIndent FirstIndent{ Log };

  // TODO: the checks should be enabled conditionally based on the user.
  revng::NonDebugInfoCheckingIRBuilder Builder(F.getContext());
  const DebugLoc &PHIDebugLoc = (*PHIs.begin())->getDebugLoc();
  Builder.SetInsertPointPastAllocas(&F, PHIDebugLoc);

  AllocaInst *Alloca = Builder.CreateAlloca((*PHIs.begin())->getType());
  revng_log(Log, "Created Alloca: " << dumpToString(Alloca));

  {
    // First, we replace all the incoming that are not internal to the
    // equivalence class with stores of the incoming in the associated local
    // variable. This may not be always possible, in which case we have to add
    // additional BasicBlocks.
    revng_log(Log, "Replacing Incomings");
    LoggerIndent IndentIncomings{ Log };

    auto IncomingUsesOfValueFromBlock = getIncomingUsesOfValuesFromBlocks(PHIs);

    auto BlockIt = IncomingUsesOfValueFromBlock.begin();
    auto BlockNext = IncomingUsesOfValueFromBlock.begin();
    auto BlockEnd = IncomingUsesOfValueFromBlock.end();

    // Handy helper to advance the iterators, so that the range from BlockIt to
    // BlockNext always contains entries that belong to the same Block.
    const auto AdvanceBlockRange = [&BlockIt, &BlockNext, &BlockEnd]() {
      BlockIt = BlockNext;
      if (BlockIt != BlockEnd) {
        BasicBlock *NewBlock = BlockIt->first.first;

        const auto IsSameBlock = [NewBlock](const auto &BlockAndValueUses) {
          auto *Block = BlockAndValueUses.first.first;
          return Block == NewBlock;
        };

        BlockNext = std::find_if_not(BlockIt, BlockEnd, IsSameBlock);
      }
      return BlockIt;
    };

    while (AdvanceBlockRange() != BlockEnd) {
      auto SameBlockValueUses = llvm::make_range(BlockIt, BlockNext);
      revng_assert(not SameBlockValueUses.empty());
      // We handle the first element in the range separately, since it's the one
      // with the highest number of uses.
      const BasicBlock *CurrentBlock = nullptr;
      {
        auto &[BlockAndValue, IncomingUses] = *SameBlockValueUses.begin();
        auto &[IncomingBlock, Incoming] = BlockAndValue;
        revng_log(Log, "IncomingBlock: " << IncomingBlock->getName());
        revng_log(Log, "Incoming: " << dumpToString(Incoming));
        buildStore(IncomingBlock, Incoming, Alloca);
        CurrentBlock = IncomingBlock;
      }

      SmallMap<std::pair<BasicBlock *, BasicBlock *>, Value *, 4> HandledCases;

      for (auto &[BlockAndValue, IncomingUses] :
           llvm::drop_begin(SameBlockValueUses)) {
        auto &[IncomingBlock, Incoming] = BlockAndValue;
        revng_log(Log, "IncomingBlock: " << IncomingBlock->getName());
        revng_log(Log, "Incoming: " << dumpToString(Incoming));
        revng_assert(IncomingBlock == CurrentBlock);
        // For all the entries after the first, we cannot inject the Store in
        // the same Block as CurrentBlock, because they would conflict with the
        // other we've just inserted.
        // Hence we have to create a new BasicBlock from Block to the proper
        // PHI, where we will inject the Store.

        LoggerIndent UsesIndent{ Log };
        for (Use *U : IncomingUses) {
          // This is the block where a PHI uses the Incoming.
          PHINode *PHIUser = cast<PHINode>(U->getUser());
          revng_log(Log, "PHIUser: " << dumpToString(PHIUser));
          BasicBlock *PHIBlock = PHIUser->getParent();
          revng_log(Log, "PHIBlock: " << PHIBlock->getName());

          // If we have 2 PHINodes in the same PHIBlock that belong to the same
          // class, we don't want to process them twice, since they must have
          // the same Incoming (because of how classes are constructed), so the
          // Store is already in place, and we don't want two of them.

          auto BlockToPHIBlock = std::make_pair(IncomingBlock, PHIBlock);
          auto BlocksToIncoming = std::make_pair(std::move(BlockToPHIBlock),
                                                 Incoming);
          const auto &[It,
                       New] = HandledCases.insert(std::move(BlocksToIncoming));
          if (not New) {
            revng_log(Log, "Already handled for this equivalence class");
            revng_assert(It->second == Incoming);
            continue;
          }

          // The incoming block cannot have only a single successor, because
          // that would mean that the whole PHIs equivalence class may only have
          // a single incoming from that block, which means we should have
          // already handled it.
          revng_assert(nullptr == IncomingBlock->getSingleSuccessor());
          BasicBlock *StoreBlock = nullptr;
          if (auto It = NewBlocks.find(BlockToPHIBlock);
              It != NewBlocks.end()) {
            revng_log(Log,
                      "New Block for store was already created for a previous "
                      "equivalence class");
            StoreBlock = It->second;
          } else {
            // Create a new block that jumps to the PHIBlock
            revng_log(Log, "New block");
            StoreBlock = BasicBlock::Create(PHIBlock->getContext(),
                                            Twine(IncomingBlock->getName())
                                              + "-to-" + PHIBlock->getName(),
                                            PHIBlock->getParent());
            NewBlocks[BlockToPHIBlock] = StoreBlock;
            Builder.SetInsertPoint(StoreBlock);
            auto *Br = Builder.CreateBr(PHIBlock);
            Br->setDebugLoc(IncomingBlock->getTerminator()->getDebugLoc());

            // Now, all the branches going from the IncomingBlock to the old
            // PHIBlock should be redirected to the StoreBlock, so they see the
            // Store.
            IncomingBlock->getTerminator()->replaceUsesOfWith(PHIBlock,
                                                              StoreBlock);
            // Also, all the incoming blocks that came from IncomingBlock so
            // they come from StoreBlock.
            PHIUser->replaceIncomingBlockWith(IncomingBlock, StoreBlock);
          }
          buildStore(StoreBlock, Incoming, Alloca);
        }
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
      NewLoad->setDebugLoc(PHIDebugLoc);
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

  revng_log(Log, "ExitSSA on: " << F.getName());
  LoggerIndent Indent{ Log };

  // A vector containing sets of equivalence classes of PHINodes.
  // Each equivalence class is composed of connected PHINodes that can form
  // trees, a DAGs, or even loops.
  // Informally, all the PHINodes in a group hold the same value, and we want to
  // create a single local variable for each DAG.
  const auto PHIClasses = getPHIEquivalenceClasses(F);
  EdgeToNewBlockMap NewBlocks;
  for (const auto &PHIGroup : PHIClasses)
    replacePHIEquivalenceClass(PHIGroup, F, NewBlocks);

  return not PHIClasses.empty();
}

char ExitSSAPass::ID = 0;

static RegisterPass<ExitSSAPass> X("exit-ssa",
                                   "Transformation pass that exits from Static "
                                   "Single Assignment form, promoting PHINodes "
                                   "to sets of Allocas, Load and Stores");
