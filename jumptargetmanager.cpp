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
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/raw_ostream.h"

// Local includes
#include "debug.h"
#include "revamb.h"
#include "ir-helpers.h"
#include "jumptargetmanager.h"

using namespace llvm;

static uint64_t getConst(Value *Constant) {
  return cast<ConstantInt>(Constant)->getLimitedValue();
}

char TranslateDirectBranchesPass::ID = 0;

static RegisterPass<TranslateDirectBranchesPass> X("translate-db",
                                                   "Translate Direct Branches"
                                                   " Pass",
                                                   false,
                                                   false);

void TranslateDirectBranchesPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<DominatorTreeWrapperPass>();
}

bool TranslateDirectBranchesPass::runOnFunction(Function &F) {
  auto& Context = F.getParent()->getContext();

  Function *ExitTB = JTM->exitTB();
  auto I = ExitTB->use_begin();
  while (I != ExitTB->use_end()) {
    // Take not of the use and increment the iterator immediately: this allows us
    // to erase the call to exit_tb without unexpecte behaviors.
    Use& ExitTBUse = *I++;
    if (auto Call = dyn_cast<CallInst>(ExitTBUse.getUser())) {
      if (Call->getCalledFunction() == ExitTB) {
        // Look for the last write to the PC
        StoreInst *PCWrite = JTM->getPrevPCWrite(Call);

        // Is destination a constant?
        ConstantInt *Address = nullptr;
        if (PCWrite != nullptr &&
            (Address = dyn_cast<ConstantInt>(PCWrite->getValueOperand()))) {
          // Compute the actual PC and get the associated BasicBlock
          uint64_t TargetPC = Address->getSExtValue();
          BasicBlock *TargetBlock = JTM->getBlockAt(TargetPC);

          // Remove unreachable right after the exit_tb
          BasicBlock::iterator I = Call;
          BasicBlock::iterator BlockEnd = Call->getParent()->end();
          assert(++I != BlockEnd && isa<UnreachableInst>(&*I));
          I->eraseFromParent();

          // Cleanup of what's afterwards (only a unconditional jump is
          // allowed)
          I = Call;
          BlockEnd = Call->getParent()->end();
          if (++I != BlockEnd)
            purgeBranch(I);

          if (TargetBlock != nullptr) {
            // A target was found, jump there
            BranchInst::Create(TargetBlock, Call);
          } else {
            // We're jumping to an invalid location, abort everything
            // TODO: emit a warning
            CallInst::Create(F.getParent()->getFunction("abort"), { }, Call);
            new UnreachableInst(Context, Call);
          }
          Call->eraseFromParent();
          PCWrite->eraseFromParent();
        }
      } else
        llvm_unreachable("Unexpected instruction using the PC");
    } else
      llvm_unreachable("Unhandled usage of the PC");
  }

  return true;
}

uint64_t TranslateDirectBranchesPass::getNextPC(Instruction *TheInstruction) {
  DominatorTree& DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  BasicBlock *Block = TheInstruction->getParent();
  BasicBlock::reverse_iterator It(TheInstruction);

  while (true) {
    BasicBlock::reverse_iterator Begin(Block->rend());

    // Go back towards the beginning of the basic block looking for a call to
    // newpc
    CallInst *Marker = nullptr;
    for (; It != Begin; It++) {
      if ((Marker = dyn_cast<CallInst>(&*It))) {
        // TODO: comparing strings is not very elegant
        if (Marker->getCalledFunction()->getName() == "newpc") {
          uint64_t PC = getConst(Marker->getArgOperand(0));
          uint64_t Size = getConst(Marker->getArgOperand(1));
          assert(Size != 0);
          return PC + Size;
        }
      }
    }

    auto *Node = DT.getNode(Block);
    assert(Node != nullptr &&
           "BasicBlock not in the dominator tree, is it reachable?" );

    Block = Node->getIDom()->getBlock();
    It = Block->rbegin();
  }

  llvm_unreachable("Can't find the PC marker");
}

char JumpTargetsFromConstantsPass::ID = 0;

bool JumpTargetsFromConstantsPass::runOnFunction(Function &F) {
  for (BasicBlock& BB : make_range(F.begin(), F.end()))
    if (Visited.find(&BB) == Visited.end()) {
      Visited.insert(&BB);

      std::stack<User *> WorkList;

      // Use a lambda so we don't have to initialize the queue with all the
      // instructions
      auto Process = [this, &WorkList] (User *U) {
        auto *Call = dyn_cast<CallInst>(U);
        // TODO: comparing strings is not very elegant
        if (Call != nullptr && Call->getCalledFunction()->getName() == "newpc")
          return;

        auto *Store = dyn_cast<StoreInst>(U);
        if (Store != nullptr && JTM->isPCReg(Store->getPointerOperand()))
          return;

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

JumpTargetManager::JumpTargetManager(Function *TheFunction,
                                     Value *PCReg,
                                     Architecture& SourceArchitecture,
                                     std::vector<SegmentInfo>& Segments) :
  TheModule(*TheFunction->getParent()),
  Context(TheModule.getContext()),
  TheFunction(TheFunction),
  OriginalInstructionAddresses(),
  JumpTargets(),
  PCReg(PCReg),
  ExitTB(nullptr),
  Dispatcher(nullptr),
  DispatcherSwitch(nullptr) {
  FunctionType *ExitTBTy = FunctionType::get(Type::getVoidTy(Context),
                                             { },
                                             false);
  ExitTB = cast<Function>(TheModule.getOrInsertFunction("exitTB", ExitTBTy));
  createDispatcher(TheFunction, PCReg, true);

  for (auto& Segment : Segments) {
    if (Segment.IsExecutable) {
      ExecutableRanges.push_back(std::make_pair(Segment.StartVirtualAddress,
                                                Segment.EndVirtualAddress));
    }

    auto *Data = cast<ConstantDataArray>(Segment.Variable->getInitializer());
    const unsigned char *DataStart = Data->getRawDataValues().bytes_begin();
    const unsigned char *DataEnd = Data->getRawDataValues().bytes_end();

    if (SourceArchitecture.pointerSize() == 64) {
      if (SourceArchitecture.isLittleEndian())
        findCodePointers<uint64_t, support::endianness::little>(DataStart,
                                                                DataEnd);
      else
        findCodePointers<uint64_t, support::endianness::big>(DataStart,
                                                             DataEnd);
    } else if (SourceArchitecture.pointerSize() == 32) {
      if (SourceArchitecture.isLittleEndian())
        findCodePointers<uint32_t, support::endianness::little>(DataStart,
                                                                DataEnd);
      else
        findCodePointers<uint32_t, support::endianness::big>(DataStart,
                                                             DataEnd);
    }
  }
}

template<typename value_type, unsigned endian>
void JumpTargetManager::findCodePointers(const unsigned char *Start,
                                         const unsigned char *End) {
  using support::endian::read;
  using support::endianness;
  for (; Start < End - sizeof(value_type); Start++) {
    uint64_t Value = read<value_type, static_cast<endianness>(endian), 1>(Start);
    getBlockAt(Value);
  }
}

TranslateDirectBranchesPass
*JumpTargetManager::createTranslateDirectBranchesPass() {
  return new TranslateDirectBranchesPass(this);
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

  // TODO: handle the following case:
  //          pc = x
  //          brcond ?, a, b
  //       a:
  //          pc = y
  //          br b
  //       b:
  //          exitTB
  // TODO: emit warning
  return nullptr;
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
        BasicBlock::iterator I = Call;
        BasicBlock::iterator BlockEnd = Call->getParent()->end();
        assert(++I != BlockEnd && isa<UnreachableInst>(&*I));
        I->eraseFromParent();
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
