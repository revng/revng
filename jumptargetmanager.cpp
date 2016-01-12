/// \file
/// \brief This file handles the possible jump targets encountered during
///        translation and the creation and management of the respective
///        BasicBlock.

// Standard includes
#include <cstdint>
#include <sstream>
#include <stack>

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

char JumpTargetsFromConstantsPass::ID = 0;

bool JumpTargetsFromConstantsPass::runOnFunction(Function &F) {
  for (BasicBlock& BB : make_range(F.begin(), F.end()))
    if (Visited.find(&BB) == Visited.end()) {
      Visited.insert(&BB);

      std::stack<User *> WorkList;

      // Use a lambda so we don't have to initialize the queue with all the
      // instructions
      auto Process = [this, &WorkList] (User *U) {
        for (Use& Operand : U->operands()) {
          auto *OperandUser = dyn_cast<User>(Operand.get());
          if (OperandUser != nullptr
              && OperandUser->op_begin() != OperandUser->op_end()) {
            WorkList.push(OperandUser);
          }

          auto *Constant = dyn_cast<ConstantInt>(Operand.get());
          if (Constant != nullptr)
            JTM->getBlockAt(Constant->getLimitedValue());

        }

      };

      for (Instruction& Instr : BB)
        Process(&Instr);

      while (!WorkList.empty()) {
        auto *Current = WorkList.top();
        WorkList.pop();
        Process(Current);
      }
    }
  return false;
}

JumpTargetManager::JumpTargetManager(Module& TheModule,
                                     Value *PCReg,
                                     Function *TheFunction,
                                     RangesVector& ExecutableRanges) :
  TheModule(TheModule),
  Context(TheModule.getContext()),
  TheFunction(TheFunction),
  OriginalInstructionAddresses(),
  JumpTargets(),
  PCReg(PCReg),
  ExitTB(nullptr),
  ExecutableRanges(ExecutableRanges),
  Dispatcher(nullptr),
  DispatcherSwitch(nullptr) {
  FunctionType *ExitTBTy = FunctionType::get(Type::getVoidTy(Context),
                                             { },
                                             false);
  ExitTB = cast<Function>(TheModule.getOrInsertFunction("exitTB", ExitTBTy));
  createDispatcher(TheFunction, PCReg, true);
}

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
// TODO: make this return a pair
BasicBlock *JumpTargetManager::newPC(uint64_t PC, bool& ShouldContinue) {
  // Did we already meet this PC?
  auto JTIt = JumpTargets.find(PC);
  if (JTIt != JumpTargets.end()) {
    // If it was planned to explore it in the future, just to do it now
    for (auto UnexploredIt = Unexplored.begin();
         UnexploredIt != Unexplored.end();
         UnexploredIt++) {

      if (UnexploredIt->first == PC) {
        auto Result = UnexploredIt->second;
        Unexplored.erase(UnexploredIt);
        ShouldContinue = true;
        assert(Result->empty());
        return Result;
      }

    }

    // It wasn't planned to visit it, so we've already been there, just jump
    // there
    assert(!JTIt->second->empty());
    ShouldContinue = false;
    return JTIt->second;
  }

  // Check if already translated this PC even if it's not associated to a basic
  // block. This typically happens with variable-length instruction encodings.
  auto OIAIt = OriginalInstructionAddresses.find(PC);
  if (OIAIt != OriginalInstructionAddresses.end()) {
    ShouldContinue = false;
    return getBlockAt(PC);
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

StoreInst *JumpTargetManager::getPrevPCWrite(Instruction *TheInstruction) {
  // Look for the last write to the PC
  BasicBlock::iterator I(TheInstruction);
  BasicBlock::iterator Begin(TheInstruction->getParent()->begin());

  while (I != Begin) {
    I--;
    Instruction *Current = &*I;

    auto *Store = dyn_cast<StoreInst>(Current);
    if (Store != nullptr && Store->getPointerOperand() == PCReg)
      return Store;

    // If we meet a call to an helper, return nullptr
    // TODO: for now we just make calls to helpers, is this is OK even if we
    //       split the translated function in multiple functions?
    if (isa<CallInst>(Current))
      return nullptr;
  }

  assert(false &&
         "Couldn't find a write to the PC in the basic block of an exit_tb");
}

void JumpTargetManager::translateIndirectJumps() {
  if (ExitTB->use_empty())
    return;

  auto I = ExitTB->use_begin();
  while (I != ExitTB->use_end()) {
    Use& ExitTBUse = *I++;
    if (auto Call = dyn_cast<CallInst>(ExitTBUse.getUser())) {
      if (Call->getCalledFunction() == ExitTB) {
        // Look for the last write to the PC
        StoreInst *Jump = getPrevPCWrite(Call);
        assert((Jump == nullptr ||
                !isa<ConstantInt>(Jump->getValueOperand()))
               && "Direct jumps should not be handled here");

        BasicBlock *BB = Call->getParent();
        auto *Branch = BranchInst::Create(Dispatcher, Call);
        Call->eraseFromParent();

        // Cleanup everything it's aftewards
        Instruction *ToDelete = &*(--BB->end());
        while (ToDelete != Branch) {
          if (auto DeadBranch = dyn_cast<BranchInst>(ToDelete))
            purgeBranch(DeadBranch);
          else
            ToDelete->eraseFromParent();

          ToDelete = &*(--BB->end());
        }
      }
    }
  }
}

JumpTargetManager::BlockWithAddress JumpTargetManager::peek() {
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
  if (!isExecutableAddress(PC)) {
    assert("Jump to a non-executable address");
    return nullptr;
  }

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
      NewBlock = ContainingBlock->splitBasicBlock(InstrIt->second);
    }
  } else {
    // Case 3: the address has never been met, create a temporary one, register
    // it for future exploration and return it
    std::stringstream Name;
    Name << "bb.0x" << std::hex << PC;

    NewBlock = BasicBlock::Create(Context, Name.str(), TheFunction);
    Unexplored.push_back(BlockWithAddress(PC, NewBlock));

    // Create a case for the address associated to the new block
    auto *PCRegType = PCReg->getType();
    auto *SwitchType = cast<IntegerType>(PCRegType->getPointerElementType());
    DispatcherSwitch->addCase(ConstantInt::get(SwitchType, PC), NewBlock);
  }

  // Associate the PC with the chosen basic block
  JumpTargets[PC] = NewBlock;
  return NewBlock;
}

// TODO: instead of a gigantic switch case we could map the original memory area
//       and write the address of the translated basic block at the jump target
// If this function looks weird it's because it has been designed to be able
// to create the dispatcher in the "root" function or in a standalone function
void JumpTargetManager::createDispatcher(Function *OutputFunction,
                                         Value *SwitchOnPtr,
                                         bool JumpDirectly) {
  IRBuilder<> Builder(Context);

  // Create the first block of the dispatcher
  BasicBlock *Entry = BasicBlock::Create(Context,
                                         "dispatcher.entry",
                                         OutputFunction);

  // The default case of the switch statement it's an unhandled cases
  auto *Default = BasicBlock::Create(Context,
                                     "dispatcher.default",
                                     OutputFunction);
  Builder.SetInsertPoint(Default);
  Builder.CreateCall(TheFunction->getParent()->getFunction("abort"));
  Builder.CreateUnreachable();

  // Switch on the first argument of the function
  Builder.SetInsertPoint(Entry);
  Value *SwitchOn = Builder.CreateLoad(SwitchOnPtr);
  SwitchInst *Switch = Builder.CreateSwitch(SwitchOn, Default);
  auto *SwitchOnType = cast<IntegerType>(SwitchOn->getType());

  {
    // We consider a jump to NULL as a program end
    auto *NullBlock = BasicBlock::Create(Context,
                                         "dispatcher.case.null",
                                         OutputFunction);
    Switch->addCase(ConstantInt::get(SwitchOnType, 0), NullBlock);
    Builder.SetInsertPoint(NullBlock);
    Builder.CreateRetVoid();
  }

  Dispatcher = Entry;
  DispatcherSwitch = Switch;
}

const JumpTargetManager::BlockWithAddress JumpTargetManager::NoMoreTargets =
  JumpTargetManager::BlockWithAddress(0, nullptr);
