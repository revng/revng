/// \file
/// \brief This file handles the possible jump targets encountered during
///        translation and the creation and management of the respective
///        BasicBlock.

// Standard includes
#include <cstdint>

// LLVM includes
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"

// Local includes
#include "ir-helpers.h"
#include "jumptargetmanager.h"

using namespace llvm;

JumpTargetManager::JumpTargetManager(Module& TheModule,
                                     Value *PCReg,
                                     Function *TheFunction) :
  TheModule(TheModule),
  Context(TheModule.getContext()),
  TheFunction(TheFunction),
  OriginalInstructionAddresses(),
  JumpTargets(),
  PCReg(PCReg) { }

/// Handle a new program counter. We might already have a basic block for that
/// program counter, or we could even have a translation for it. Return one
/// of these, if appropriate.
///
/// \param PC the new program counter.
/// \param ShouldContinue an out parameter indicating whether the returned
///        basic block was just a placeholder or actually contains a
///        translation.
///
/// \return the basic block to use from now on, or null if the program counter
///         is not associated to a basic block.
BasicBlock *JumpTargetManager::newPC(uint64_t PC, bool& ShouldContinue) {
  // Did we already meet this PC?
  auto It = JumpTargets.find(PC);
  if (It != JumpTargets.end()) {
    // If it was planned to explore it in the future, just to do it now
    for (auto It = Unexplored.begin(); It != Unexplored.end(); It++) {
      if (It->first == PC) {
        Unexplored.erase(It, It + 1);
        ShouldContinue = true;
        assert(It->second->empty());
        return It->second;
      }
    }

    // It wasn't planned to visit it, so we've already been there, just jump
    // there
    assert(!It->second->empty());
    ShouldContinue = false;
    return It->second;
  }

  // We don't know anything about this PC
  return nullptr;
}

/// Save the PC-Instruction association for future use (jump target)
void JumpTargetManager::registerInstruction(uint64_t PC,
                                            Instruction *Instruction) {
  // Never save twice a PC
  assert(OriginalInstructionAddresses.find(PC) ==
         OriginalInstructionAddresses.end());
  OriginalInstructionAddresses[PC] = Instruction;
}

/// Save the PC-BasicBlock association for futur use (jump target)
void JumpTargetManager::registerBlock(uint64_t PC, BasicBlock *Block) {
  // If we already met it, it must point to the same block
  auto It = JumpTargets.find(PC);
  assert(It == JumpTargets.end() || It->second == Block);
  if (It->second != Block)
    JumpTargets[PC] = Block;
}

void JumpTargetManager::translateIndirectJumps() {
  BasicBlock *Dispatcher = createDispatcher(TheFunction, PCReg, true);

  for (Use& PCUse : PCReg->uses()) {
    if (PCUse.getOperandNo() == 1) {
      if (auto Jump = dyn_cast<StoreInst>(PCUse.getUser())) {
        BasicBlock::iterator It(Jump);
        auto *Branch = BranchInst::Create(Dispatcher, ++It);

        // Cleanup everything it's aftewards
        BasicBlock *Parent = Jump->getParent();
        Instruction *ToDelete = &*(--Parent->end());
        while (ToDelete != Branch) {
          if (auto DeadBranch = dyn_cast<BranchInst>(ToDelete))
            purgeBranch(DeadBranch);
          else
            ToDelete->eraseFromParent();

          ToDelete = &*(--Parent->end());
        }
      }
    }
  }
}

Value *JumpTargetManager::PC() {
  return PCReg;
}

/// Pop from the list of program counters to explore
///
/// \return a pair containing the PC and the initial block to use, or
///         JumpTarget::NoMoreTargets if we're done.
JumpTargetManager::BlockWithAddress JumpTargetManager::peekJumpTarget() {
  if (Unexplored.empty())
    return NoMoreTargets;
  else {
    BlockWithAddress Result = Unexplored.back();
    Unexplored.pop_back();
    return Result;
  }
}

/// Get or create a block for the given PC
BasicBlock *JumpTargetManager::getBlockAt(uint64_t PC) {
  // Do we already have a BasicBlock for this PC?
  BlockMap::iterator TargetIt = JumpTargets.find(PC);
  if (TargetIt != JumpTargets.end()) {
    // Case 1: there's already a BasicBlock for that address, return it
    return TargetIt->second;
  }

  // Did we already meet this PC (i.e. do we know what's the associated
  // instruction)?
  BasicBlock *NewBlock = nullptr;
  InstructionMap::iterator InstrIt = OriginalInstructionAddresses.find(PC);
  if (InstrIt != OriginalInstructionAddresses.end()) {
    // Case 2: the address has already been met, but needs to be promoted to
    //         BasicBlock level.
    BasicBlock *ContainingBlock = InstrIt->second->getParent();
    if (InstrIt->second == &*ContainingBlock->begin())
      NewBlock = ContainingBlock;
    else {
      assert(InstrIt->second != nullptr
             && InstrIt->second != ContainingBlock->end());
      // Split the block in the appropriate position. Note that
      // OriginalInstructionAddresses stores a reference to the last generated
      // instruction for the previous instruction.
      // We add an llvm_unreachable just to be able to split the basic block
      Instruction *Next = InstrIt->second->getNextNode();
      Instruction *Terminator = new UnreachableInst(Context, ContainingBlock);
      NewBlock = ContainingBlock->splitBasicBlock(Next);
      Terminator->eraseFromParent();
    }
  } else {
    // Case 3: the address has never been met, create a temporary one, register
    // it for future exploration and return it
    NewBlock = BasicBlock::Create(Context, "", TheFunction);
    Unexplored.push_back(BlockWithAddress(PC, NewBlock));
  }

  // Associate the PC with the chosen basic block
  JumpTargets[PC] = NewBlock;
  return NewBlock;
}

// TODO: instead of a gigantic switch case we could map the original memory area
//       and write the address of the translated basic block at the jump target
BasicBlock *JumpTargetManager::createDispatcher(Function *OutputFunction,
                                                Value *SwitchOnPtr,
                                                bool JumpDirectly) {
  IRBuilder<> Builder(Context);

  // Create the first block of the function
  BasicBlock *Entry = BasicBlock::Create(Context, "", OutputFunction);

  // The default case of the switch statement it's an unhandled cases
  auto *Default = BasicBlock::Create(Context, "", OutputFunction);
  Builder.SetInsertPoint(Default);
  Builder.CreateUnreachable();

  // Switch on the first argument of the function
  Builder.SetInsertPoint(Entry);
  Value *SwitchOn = Builder.CreateLoad(SwitchOnPtr);
  SwitchInst *Switch = Builder.CreateSwitch(SwitchOn, Default);
  auto *SwitchOnType = cast<IntegerType>(SwitchOn->getType());

  {
    // We consider a jump to NULL as a program end
    auto *NullBlock = BasicBlock::Create(Context, "", OutputFunction);
    Switch->addCase(ConstantInt::get(SwitchOnType, 0), NullBlock);
    Builder.SetInsertPoint(NullBlock);
    Builder.CreateRetVoid();
  }

  // Create a case for each jump target we saw so far
  for (auto& Pair : JumpTargets) {
    // Create a case for the address associated to the current block
    auto *Block = BasicBlock::Create(Context, "", OutputFunction);
    Switch->addCase(ConstantInt::get(SwitchOnType, Pair.first), Block);

    Builder.SetInsertPoint(Block);
    if (JumpDirectly) {
      // Assume we're injecting the switch case directly into the function
      // the blocks are in, so we can jump to the target block directly
      assert(Pair.second->getParent() == OutputFunction);
      Builder.CreateBr(Pair.second);
    } else {
      // Return the address of the current block
      Builder.CreateRet(BlockAddress::get(OutputFunction, Pair.second));
    }
  }

  return Entry;
}

const JumpTargetManager::BlockWithAddress JumpTargetManager::NoMoreTargets =
  JumpTargetManager::BlockWithAddress(0, nullptr);
