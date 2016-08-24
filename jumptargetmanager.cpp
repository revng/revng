/// \file
/// \brief This file handles the possible jump targets encountered during
///        translation and the creation and management of the respective
///        BasicBlock.

// Standard includes
#include <cassert>
#include <cstdint>
#include <fstream>
#include <queue>
#include <sstream>

// LLVM includes
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Endian.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"

// Local includes
#include "debug.h"
#include "revamb.h"
#include "ir-helpers.h"
#include "jumptargetmanager.h"
#include "set.h"
#include "simplifycomparisons.h"

using namespace llvm;

static bool isSumJump(StoreInst *PCWrite);

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
  AU.addUsedIfAvailable<SETPass>();
}

/// \brief Purges everything is after a call to exitTB (except the call itself)
static void exitTBCleanup(Instruction *ExitTBCall) {
  BasicBlock *BB = ExitTBCall->getParent();
  BasicBlock::iterator BlockEnd(BB->end());

  // Cleanup everything it's aftewards starting from the end
  Instruction *ToDelete = &*(--BB->end());
  while (ToDelete != ExitTBCall) {
    if (auto DeadBranch = dyn_cast<BranchInst>(ToDelete))
      purgeBranch(BasicBlock::iterator(DeadBranch));
    else
      ToDelete->eraseFromParent();

    ToDelete = &*(--BB->end());
  }
}

bool TranslateDirectBranchesPass::pinJTs(Function &F) {
  const auto *SET = getAnalysisIfAvailable<SETPass>();
  if (SET == nullptr || SET->jumps().size() == 0)
    return false;

  auto& Context = F.getParent()->getContext();
  auto *PCReg = JTM->pcReg();
  auto *RegType = cast<IntegerType>(PCReg->getType()->getPointerElementType());
  auto C = [RegType] (uint64_t A) { return ConstantInt::get(RegType, A); };
  BasicBlock *Dispatcher = JTM->dispatcher();
  BasicBlock *DispatcherFail = JTM->dispatcherFail();

  for (const auto &Jump : SET->jumps()) {
    StoreInst *PCWrite = Jump.Instruction;
    bool Approximate = Jump.Approximate;
    const std::vector<uint64_t> &Destinations = Jump.Destinations;

    // We don't care if we already handled this call too exitTB in the past,
    // information should become progressively more precise, so let's just
    // remove everything after this call and put a new handler
    CallInst *CallExitTB = JTM->findNextExitTB(PCWrite);

    assert(CallExitTB != nullptr);
    assert(PCWrite->getParent()->getParent() == &F);
    assert(JTM->isPCReg(PCWrite->getPointerOperand()));
    assert(Destinations.size() != 0);

    auto *ExitTBArg = ConstantInt::get(Type::getInt32Ty(Context),
                                       Destinations.size());
    uint64_t OldTargetsCount = getLimitedValue(CallExitTB->getArgOperand(0));

    // TODO: we should check Destinations.size() >= OldTargetsCount
    // TODO: we should also check the destinations are actually the same

    BasicBlock *FailBB = Approximate ? Dispatcher : Dispatcher /* Fail */;
    BasicBlock *BB = CallExitTB->getParent();

    // Kill everything is after the call to exitTB
    exitTBCleanup(CallExitTB);

    // Mark this call to exitTB as handled
    CallExitTB->setArgOperand(0, ExitTBArg);

    IRBuilder<> Builder(BB);
    auto PCLoad = Builder.CreateLoad(PCReg);
    if (Destinations.size() == 1) {
      auto *Comparison = Builder.CreateICmpEQ(C(Destinations[0]), PCLoad);
      Builder.CreateCondBr(Comparison,
                           JTM->getBlockAt(Destinations[0]),
                           FailBB);
    } else {
      auto *Switch = Builder.CreateSwitch(PCLoad, FailBB, Destinations.size());
      for (uint64_t Destination : Destinations)
        Switch->addCase(C(Destination), JTM->getBlockAt(Destination));
    }

    // Notify new branches only if the amount of possible targets actually
    // increased
    if (Destinations.size() > OldTargetsCount)
      JTM->newBranch();
  }

  return true;
}

bool TranslateDirectBranchesPass::pinConstantStore(Function &F) {
  auto& Context = F.getParent()->getContext();

  Function *ExitTB = JTM->exitTB();
  auto ExitTBIt = ExitTB->use_begin();
  while (ExitTBIt != ExitTB->use_end()) {
    // Take note of the use and increment the iterator immediately: this allows
    // us to erase the call to exit_tb without unexpected behaviors
    Use& ExitTBUse = *ExitTBIt++;
    if (auto Call = dyn_cast<CallInst>(ExitTBUse.getUser())) {
      if (Call->getCalledFunction() == ExitTB) {
        // Look for the last write to the PC
        StoreInst *PCWrite = JTM->getPrevPCWrite(Call);

        // Is destination a constant?
        if (PCWrite == nullptr) {
          forceFallthroughAfterHelper(Call);
        } else {
          uint64_t NextPC = JTM->getNextPC(PCWrite);
          if (NextPC != 0 && JTM->isOSRAEnabled() && isSumJump(PCWrite))
            JTM->registerJT(NextPC, JumpTargetManager::SumJump);

          auto *Address = dyn_cast<ConstantInt>(PCWrite->getValueOperand());
          if (Address != nullptr) {
            // Compute the actual PC and get the associated BasicBlock
            uint64_t TargetPC = Address->getSExtValue();
            // TODO: can we switch to getBlockAt()?
            auto *TargetBlock = JTM->registerJT(TargetPC,
                                                JumpTargetManager::DirectJump);

            // Remove unreachable right after the exit_tb
            BasicBlock::iterator CallIt(Call);
            BasicBlock::iterator BlockEnd = Call->getParent()->end();
            CallIt++;
            assert(CallIt != BlockEnd && isa<UnreachableInst>(&*CallIt));
            CallIt->eraseFromParent();

            // Cleanup of what's afterwards (only a unconditional jump is
            // allowed)
            CallIt = BasicBlock::iterator(Call);
            BlockEnd = Call->getParent()->end();
            if (++CallIt != BlockEnd)
              purgeBranch(CallIt);

            if (TargetBlock != nullptr) {
              // A target was found, jump there
              BranchInst::Create(TargetBlock, Call);
              JTM->newBranch();
            } else {
              // We're jumping to an invalid location, abort everything
              // TODO: emit a warning
              CallInst::Create(F.getParent()->getFunction("abort"), { }, Call);
              new UnreachableInst(Context, Call);
            }
            Call->eraseFromParent();
          }
        }
      } else
        llvm_unreachable("Unexpected instruction using the PC");
    } else
      llvm_unreachable("Unhandled usage of the PC");
  }

  return true;
}

bool TranslateDirectBranchesPass::forceFallthroughAfterHelper(CallInst *Call) {
  // If someone else already took care of the situation, quit
  if (getLimitedValue(Call->getArgOperand(0)) > 0)
    return false;

  auto *PCReg = JTM->pcReg();
  auto PCRegTy = PCReg->getType()->getPointerElementType();
  bool ForceFallthrough = false;

  BasicBlock::reverse_iterator It(make_reverse_iterator(Call));
  auto *BB = Call->getParent();
  auto EndIt = BB->rend();
  while (!ForceFallthrough) {
    while (It != EndIt) {
      Instruction *I = &*It;
      if (auto *Store = dyn_cast<StoreInst>(I)) {
        if (Store->getPointerOperand() == PCReg) {
          // We found a PC-store, give up
          return false;
        }
      } else if (auto *Call = dyn_cast<CallInst>(I)) {
        if (Function *Callee = Call->getCalledFunction()) {
          if (Callee->getName().startswith("helper_")) {
            // We found a call to an helper
            ForceFallthrough = true;
            break;
          }
        }
      }
      It++;
    }

    if (!ForceFallthrough) {
      // Proceed only to unique predecessor, if present
      if (auto *Pred = BB->getUniquePredecessor()) {
        BB = Pred;
        It = BB->rbegin();
        EndIt = BB->rend();
      } else {
        // We have multiple predecessors, give up
        return false;
      }
    }

  }

  exitTBCleanup(Call);
  JTM->newBranch();

  IRBuilder<> Builder(Call->getParent());
  Call->setArgOperand(0, Builder.getInt32(1));

  // Create the fallthrough jump
  uint64_t NextPC = JTM->getNextPC(Call);
  Value *NextPCConst = Builder.getIntN(PCRegTy->getIntegerBitWidth(), NextPC);
  Builder.CreateCondBr(Builder.CreateICmpEQ(Builder.CreateLoad(PCReg),
                                            NextPCConst),
                       JTM->registerJT(NextPC, JumpTargetManager::PostHelper),
                       JTM->dispatcher());

  return true;
}

bool TranslateDirectBranchesPass::runOnFunction(Function &F) {
  pinConstantStore(F);
  pinJTs(F);
  return true;
}

uint64_t TranslateDirectBranchesPass::getNextPC(Instruction *TheInstruction) {
  DominatorTree& DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  BasicBlock *Block = TheInstruction->getParent();
  BasicBlock::reverse_iterator It(make_reverse_iterator(TheInstruction));

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

Constant *JumpTargetManager::readConstantPointer(Constant *Address,
                                                 Type *PointerTy) const {
  auto *Value = readConstantInt(Address, SourceArchitecture.pointerSize());
  if (Value != nullptr) {
    return ConstantExpr::getIntToPtr(Value, PointerTy);
  } else {
    return nullptr;
  }
}

ConstantInt *JumpTargetManager::readConstantInt(Constant *ConstantAddress,
                                                unsigned Size) const {
  // TODO: register that the value has been used externally
  return readConstantInternal(ConstantAddress, Size);
}

ConstantInt *JumpTargetManager::readConstantInternal(Constant *ConstantAddress,
                                                     unsigned Size) const {
  const DataLayout &DL = TheModule.getDataLayout();

  if (ConstantAddress->getType()->isPointerTy()) {
    using CE = ConstantExpr;
    auto IntPtrTy = Type::getIntNTy(Context, SourceArchitecture.pointerSize());
    ConstantAddress = CE::getPtrToInt(ConstantAddress, IntPtrTy);
  }

  uint64_t Address = getZExtValue(ConstantAddress, DL);

  for (auto &Segment : Segments) {
    // Note: we also consider writeable memory areas because, despite being
    // modifiable, can contain useful information
    if (Segment.contains(Address, Size) && Segment.IsReadable) {
      auto *Array = cast<ConstantDataArray>(Segment.Variable->getInitializer());
      StringRef RawData = Array->getRawDataValues();
      const unsigned char *RawDataPtr = RawData.bytes_begin();
      uint64_t Offset = Address - Segment.StartVirtualAddress;
      const unsigned char *Start = RawDataPtr + Offset;

      using support::endian::read;
      using support::endianness;
      uint64_t Value;
      switch (Size) {
      case 1:
        Value = read<uint8_t, endianness::little, 1>(Start);
        break;
      case 2:
        if (DL.isLittleEndian())
          Value = read<uint16_t, endianness::little, 1>(Start);
        else
          Value = read<uint16_t, endianness::big, 1>(Start);
        break;
      case 4:
        if (DL.isLittleEndian())
          Value = read<uint32_t, endianness::little, 1>(Start);
        else
          Value = read<uint32_t, endianness::big, 1>(Start);
        break;
      case 8:
        if (DL.isLittleEndian())
          Value = read<uint64_t, endianness::little, 1>(Start);
        else
          Value = read<uint64_t, endianness::big, 1>(Start);
        break;
      default:
        llvm_unreachable("Unexpected read size");
      }

      return ConstantInt::get(IntegerType::get(Context, Size * 8), Value);
    }
  }

  return nullptr;
}

template<typename T>
static cl::opt<T> *getOption(StringMap<cl::Option *>& Options,
                             const char *Name) {
  return static_cast<cl::opt<T> *>(Options[Name]);
}

JumpTargetManager::JumpTargetManager(Function *TheFunction,
                                     Value *PCReg,
                                     Architecture& SourceArchitecture,
                                     std::vector<SegmentInfo>& Segments,
                                     bool EnableOSRA) :
  TheModule(*TheFunction->getParent()),
  Context(TheModule.getContext()),
  TheFunction(TheFunction),
  OriginalInstructionAddresses(),
  JumpTargets(),
  PCReg(PCReg),
  ExitTB(nullptr),
  Dispatcher(nullptr),
  DispatcherSwitch(nullptr),
  Segments(Segments),
  SourceArchitecture(SourceArchitecture),
  EnableOSRA(EnableOSRA) {
  FunctionType *ExitTBTy = FunctionType::get(Type::getVoidTy(Context),
                                             { Type::getInt32Ty(Context) },
                                             false);
  ExitTB = cast<Function>(TheModule.getOrInsertFunction("exitTB", ExitTBTy));
  createDispatcher(TheFunction, PCReg, true);

  for (auto& Segment : Segments)
    Segment.insertExecutableRanges(std::back_inserter(ExecutableRanges));

  // Configure GlobalValueNumbering
  StringMap<cl::Option *>& Options(cl::getRegisteredOptions());
  getOption<bool>(Options, "enable-load-pre")->setInitialValue(false);
  getOption<unsigned>(Options, "memdep-block-scan-limit")->setInitialValue(100);
  // getOption<bool>(Options, "enable-pre")->setInitialValue(false);
  // getOption<uint32_t>(Options, "max-recurse-depth")->setInitialValue(10);
}

void JumpTargetManager::harvestGlobalData() {
  for (auto& Segment : Segments) {
    auto *Data = cast<ConstantDataArray>(Segment.Variable->getInitializer());
    const unsigned char *DataStart = Data->getRawDataValues().bytes_begin();
    const unsigned char *DataEnd = Data->getRawDataValues().bytes_end();

    using endianness = support::endianness;
    if (SourceArchitecture.pointerSize() == 64) {
      if (SourceArchitecture.isLittleEndian())
        findCodePointers<uint64_t, endianness::little>(DataStart, DataEnd);
      else
        findCodePointers<uint64_t, endianness::big>(DataStart, DataEnd);
    } else if (SourceArchitecture.pointerSize() == 32) {
      if (SourceArchitecture.isLittleEndian())
        findCodePointers<uint32_t, endianness::little>(DataStart, DataEnd);
      else
        findCodePointers<uint32_t, endianness::big>(DataStart, DataEnd);
    }
  }

  DBG("jtcount", dbg
      << "JumpTargets found in global data: " << std::dec
      << Unexplored.size() << "\n");
}

template<typename value_type, unsigned endian>
void JumpTargetManager::findCodePointers(const unsigned char *Start,
                                         const unsigned char *End) {
  using support::endian::read;
  using support::endianness;
  for (; Start < End - sizeof(value_type); Start++) {
    uint64_t Value = read<value_type,
                          static_cast<endianness>(endian),
                          1>(Start);
    registerJT(Value, GlobalData);
  }
}

/// Handle a new program counter. We might already have a basic block for that
/// program counter, or we could even have a translation for it. Return one of
/// these, if appropriate.
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

  // Check if we already translated this PC even if it's not associated to a
  // basic block (i.e., we have to split its basic block). This typically
  // happens with variable-length instruction encodings.
  if (OriginalInstructionAddresses.count(PC) != 0) {
    ShouldContinue = false;
    return registerJT(PC, AmbigousInstruction);
  }

  // We don't know anything about this PC
  return nullptr;
}

/// Save the PC-Instruction association for future use (jump target)
void JumpTargetManager::registerInstruction(uint64_t PC,
                                            Instruction *Instruction) {
  // Never save twice a PC
  assert(!OriginalInstructionAddresses.count(PC));
  OriginalInstructionAddresses[PC] = Instruction;
}

CallInst *JumpTargetManager::findNextExitTB(Instruction *Start) {
  CallInst *Result = nullptr;

  visitSuccessors(Start, nullptr, [this,&Result] (BasicBlockRange Range) {
      for (Instruction &I : Range) {
        if (auto *Call = dyn_cast<CallInst>(&I)) {
          assert(!(Call->getCalledFunction()->getName() == "newpc"));
          if (Call->getCalledFunction() == ExitTB) {
            assert(Result == nullptr);
            Result = Call;
            return ExhaustQueueAndStop;
          }
        }
      }

      return Continue;
    });

  return Result;
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


/// \brief Tries to detect pc += register In general, we assume what we're
/// translating is code emitted by a compiler. This means that usually all the
/// possible jump targets are explicit jump to a constant or are stored
/// somewhere in memory (e.g.  jump tables and vtables). However, in certain
/// cases, mainly due to handcrafted assembly we can have a situation like the
/// following:
///
///     addne pc, pc, \curbit, lsl #2
///
/// (taken from libgcc ARM's lib1funcs.S, specifically line 592 of
/// `libgcc/config/arm/lib1funcs.S` at commit
/// `f1717362de1e56fe1ffab540289d7d0c6ed48b20`)
///
/// This code basically jumps forward a number of instructions depending on a
/// run-time value. Therefore, without further analysis, potentially, all the
/// coming instructions are jump targets.
///
/// To workaround this issue we use a simple heuristics, which basically
/// consists in making all the coming instructions possible jump targets until
/// the next write to the PC. In the future, we could extend this until the end
/// of the function.
static bool isSumJump(StoreInst *PCWrite) {
  // * Follow the written value recursively
  //   * Is it a `load` or a `constant`? Fine. Don't proceed.
  //   * Is it an `and`? Enqueue the operands in the worklist.
  //   * Is it an `add`? Make all the coming instructions jump targets.
  //
  // This approach has a series of problems:
  //
  // * It doesn't work with delay slots. Delay slots are handled by libtinycode
  //   as follows:
  //
  //       jump lr
  //         store btarget, lr
  //       store 3, r0
  //         store 3, r0
  //         store btarget, pc
  //
  //   Clearly, if we don't follow the loads we miss the situation we're trying
  //   to handle.
  // * It is unclear how this would perform without EarlyCSE and SROA.
  std::queue<Value *> WorkList;
  WorkList.push(PCWrite->getValueOperand());

  while (!WorkList.empty()) {
    Value *V = WorkList.front();
    WorkList.pop();

    if (isa<Constant>(V) || isa<LoadInst>(V)) {
      // Fine
    } else if (auto *BinOp = dyn_cast<BinaryOperator>(V)) {
      switch (BinOp->getOpcode()) {
      case Instruction::Add:
      case Instruction::Or:
        return true;
      case Instruction::Shl:
      case Instruction::LShr:
      case Instruction::AShr:
      case Instruction::And:
        for (auto& Operand : BinOp->operands())
          if (!isa<Constant>(Operand.get()))
            WorkList.push(Operand.get());
        break;
      default:
        // TODO: emit warning
        return false;
      }
    } else {
      // TODO: emit warning
      return false;
    }
  }

  return false;
}

std::pair<uint64_t, uint64_t>
JumpTargetManager::getPC(Instruction *TheInstruction) const {
  CallInst *NewPCCall = nullptr;
  std::set<BasicBlock *> Visited;
  std::queue<BasicBlock::reverse_iterator> WorkList;
  if (TheInstruction->getIterator() == TheInstruction->getParent()->begin())
    WorkList.push(--TheInstruction->getParent()->rend());
  else
    WorkList.push(make_reverse_iterator(TheInstruction));

  while (!WorkList.empty()) {
    auto I = WorkList.front();
    WorkList.pop();
    auto *BB = I->getParent();
    auto End = BB->rend();

    // Go through the instructions looking for calls to newpc
    for (; I != End; I++) {
      if (auto Marker = dyn_cast<CallInst>(&*I)) {
        // TODO: comparing strings is not very elegant
        auto *Callee = Marker->getCalledFunction();
        if (Callee != nullptr && Callee->getName() == "newpc") {

          // We found two distinct newpc leading to the requested instruction
          if (NewPCCall != nullptr)
            return { 0, 0 };

          NewPCCall = Marker;
          break;
        }
      }
    }

    // If we haven't find a newpc call yet, continue exploration backward
    if (NewPCCall == nullptr) {
      // If one of the predecessors is the dispatcher, don't explore any further
      for (BasicBlock *Predecessor : predecessors(BB)) {
        // Assert we didn't reach the almighty dispatcher
        assert(!(NewPCCall == nullptr && Predecessor == Dispatcher));
        if (Predecessor == Dispatcher)
          continue;
      }

      for (BasicBlock *Predecessor : predecessors(BB)) {
        // Ignore already visited or empty BBs
        if (!Predecessor->empty()
            && Visited.find(Predecessor) == Visited.end()) {
          WorkList.push(Predecessor->rbegin());
          Visited.insert(Predecessor);
        }
      }
    }

  }

  // Couldn't find the current PC
  if (NewPCCall == nullptr)
    return { 0, 0 };

  uint64_t PC = getConst(NewPCCall->getArgOperand(0));
  uint64_t Size = getConst(NewPCCall->getArgOperand(1));
  assert(Size != 0);
  return { PC, Size };
}

void JumpTargetManager::handleSumJump(Instruction *SumJump) {
  // Take the next PC
  uint64_t NextPC = getNextPC(SumJump);
  assert(NextPC != 0);
  BasicBlock *BB = registerJT(NextPC, JumpTargetManager::SumJump);
  assert(BB && !BB->empty());

  std::set<BasicBlock *> Visited;
  Visited.insert(Dispatcher);
  std::queue<BasicBlock *> WorkList;
  WorkList.push(BB);
  while (!WorkList.empty()) {
    BB = WorkList.front();
    Visited.insert(BB);
    WorkList.pop();

    BasicBlock::iterator I(BB->begin());
    BasicBlock::iterator End(BB->end());
    while (I != End) {
      // Is it a new PC marker?
      if (auto *Call = dyn_cast<CallInst>(&*I)) {
        Function *Callee = Call->getCalledFunction();
        // TODO: comparing strings is not very elegant
        if (Callee != nullptr && Callee->getName() == "newpc") {
          uint64_t PC = getConst(Call->getArgOperand(0));

          // If we've found a (direct or indirect) jump, stop
          if (PC != NextPC)
            return;

          // Split and update iterators to proceed
          BB = registerJT(PC, JumpTargetManager::SumJump);

          // Do we have a block?
          if (BB == nullptr)
            return;

          I = BB->begin();
          End = BB->end();

          // Updated the expectation for the next PC
          NextPC = PC + getConst(Call->getArgOperand(1));
        } else if (Call->getCalledFunction() == ExitTB) {
          // We've found an unparsed indirect jump
          return;
        }

      }

      // Proceed to next instruction
      I++;
    }

    // Inspect and enqueue successors
    for (BasicBlock *Successor : successors(BB))
      if (Visited.find(Successor) == Visited.end())
        WorkList.push(Successor);

  }
}

/// \brief Class to iterate over all the BBs associated to a translated PC
class BasicBlockVisitor {
public:
  BasicBlockVisitor(const SwitchInst *Dispatcher) :
    Dispatcher(Dispatcher),
    JumpTargetIndex(0),
    JumpTargetsCount(Dispatcher->getNumSuccessors()),
    DL(Dispatcher->getParent()->getParent()->getParent()->getDataLayout()) { }

  void enqueue(BasicBlock *BB) {
    if (Visited.count(BB))
      return;
    Visited.insert(BB);

    uint64_t PC = getPC(BB);
    if (PC == 0)
      SamePC.push(BB);
    else
      NewPC.push({ BB, PC });
  }

  // TODO: this function assumes 0 is not a valid PC
  std::pair<BasicBlock *, uint64_t> pop() {
    if (!SamePC.empty()) {
      auto Result = SamePC.front();
      SamePC.pop();
      return { Result, 0 };
    } else if (!NewPC.empty()) {
      auto Result = NewPC.front();
      NewPC.pop();
      return Result;
    } else if (JumpTargetIndex < JumpTargetsCount) {
      BasicBlock *BB = Dispatcher->getSuccessor(JumpTargetIndex);
      JumpTargetIndex++;
      return { BB, getPC(BB) };
    } else {
      return { nullptr, 0 };
    }
  }

private:
  // TODO: this function assumes 0 is not a valid PC
  uint64_t getPC(BasicBlock *BB) {
    if (!BB->empty()) {
      if (auto *Call = dyn_cast<CallInst>(&*BB->begin())) {
        Function *Callee = Call->getCalledFunction();
        // TODO: comparing with "newpc" string is sad
        if (Callee != nullptr && Callee->getName() == "newpc") {
          Constant *PCOperand = cast<Constant>(Call->getArgOperand(0));
          return getZExtValue(PCOperand, DL);
        }
      }
    }

    return 0;
  }

private:
  const SwitchInst *Dispatcher;
  unsigned JumpTargetIndex;
  unsigned JumpTargetsCount;
  const DataLayout &DL;
  std::set<BasicBlock *> Visited;
  std::queue<BasicBlock *> SamePC;
  std::queue<std::pair<BasicBlock *, uint64_t>> NewPC;
};

void JumpTargetManager::collectBBSummary(std::string OutputPath) {
  BasicBlockVisitor BBV(DispatcherSwitch);
  uint64_t NewPC = 0;
  uint64_t PC = 0;
  BasicBlock *BB = nullptr;
  while (NewPC == 0)
    std::tie(BB, NewPC) = BBV.pop();
  BBSummary *Summary = nullptr;

  std::set<GlobalVariable *> CPUStateSet;
  std::set<Function *> FunctionsSet;
  std::set<const char *> OpcodesSet;

  while (BB != nullptr) {
    if (NewPC != 0) {
      PC = NewPC;
      auto It = containingOriginalBB(PC);
      assert(It != OriginalBBStats.end());
      Summary = &It->second;
    }

    // Update stats
    for (Instruction &I : *BB) {
      // TODO: Data dependencies
      unsigned Opcode = I.getOpcode();
      const char *OpcodeName = I.getOpcodeName();
      Summary->Opcode[OpcodeName]++;
      OpcodesSet.insert(OpcodeName);

      switch (Opcode) {
      case Instruction::Load:
        {
          auto *L = static_cast<LoadInst *>(&I);
          if (auto *State = dyn_cast<GlobalVariable>(L->getPointerOperand())) {
            CPUStateSet.insert(State);
            Summary->ReadState[State]++;
          }

          break;
        }
      case Instruction::Store:
        {
          auto *S = static_cast<StoreInst *>(&I);
          if (auto *State = dyn_cast<GlobalVariable>(S->getPointerOperand())) {
            CPUStateSet.insert(State);
            Summary->ReadState[State]++;
          }

          break;
        }
      case Instruction::Call:
        {
          auto *Call = static_cast<CallInst *>(&I);
          if (auto *F  = Call->getCalledFunction()) {
            FunctionsSet.insert(F);
            Summary->CalledFunctions[F]++;
          }

          break;
        }
      default:
        break;
      }
    }

    std::tie(BB, NewPC) = BBV.pop();
  }

  std::vector<GlobalVariable *> CPUState;
  std::copy(CPUStateSet.begin(),
            CPUStateSet.end(),
            std::back_inserter(CPUState));
  std::vector<Function *> Functions;
  std::copy(FunctionsSet.begin(),
            FunctionsSet.end(),
            std::back_inserter(Functions));
  std::vector<const char *> Opcodes;
  std::copy(OpcodesSet.begin(),
            OpcodesSet.end(),
            std::back_inserter(Opcodes));

  std::ofstream Output(OutputPath);

  Output << "address,size";
  for (GlobalVariable *V : CPUState)
    Output << ",read_" << V->getName().str();
  for (GlobalVariable *V : CPUState)
    Output << ",write_" << V->getName().str();
  for (Function *F : Functions)
    Output << ",call_" << F->getName().str();
  for (const char *OpcodeName : Opcodes)
    Output << ",opcode_" <<  OpcodeName;
  Output << "\n";

  for (auto P : OriginalBBStats) {
    Output << std::dec << P.first << ","
        << std::dec << P.second.Size;

    for (GlobalVariable *V : CPUState)
      Output << "," << std::dec << P.second.ReadState[V];
    for (GlobalVariable *V : CPUState)
      Output << "," << std::dec << P.second.WrittenState[V];
    for (Function *F : Functions)
      Output << "," << std::dec << P.second.CalledFunctions[F];
    for (const char *OpcodeName : Opcodes)
      Output << "," << P.second.Opcode[OpcodeName];

    Output << "\n";
  }

}

void JumpTargetManager::translateIndirectJumps() {
  if (ExitTB->use_empty())
    return;

  auto I = ExitTB->use_begin();
  while (I != ExitTB->use_end()) {
    Use& ExitTBUse = *I++;
    if (auto *Call = dyn_cast<CallInst>(ExitTBUse.getUser())) {
      if (Call->getCalledFunction() == ExitTB) {

        // Look for the last write to the PC
        StoreInst *PCWrite = getPrevPCWrite(Call);
        assert((PCWrite == nullptr
                || !isa<ConstantInt>(PCWrite->getValueOperand()))
               && "Direct jumps should not be handled here");

        if (PCWrite != nullptr && EnableOSRA && isSumJump(PCWrite))
          handleSumJump(PCWrite);

        if (getLimitedValue(Call->getArgOperand(0)) == 0) {
          exitTBCleanup(Call);
          BranchInst::Create(Dispatcher, Call);
        }

        Call->eraseFromParent();
      }
    }
  }
}

JumpTargetManager::BlockWithAddress JumpTargetManager::peek() {
  harvest();

  if (Unexplored.empty())
    return NoMoreTargets;
  else {
    BlockWithAddress Result = Unexplored.back();
    Unexplored.pop_back();
    return Result;
  }
}

void JumpTargetManager::unvisit(BasicBlock *BB) {
  if (Visited.find(BB) != Visited.end()) {
    std::vector<BasicBlock *> WorkList;
    WorkList.push_back(BB);

    while (!WorkList.empty()) {
      BasicBlock *Current = WorkList.back();
      WorkList.pop_back();

      Visited.erase(Current);

      for (BasicBlock *Successor : successors(BB)) {
        if (Visited.find(Successor) != Visited.end()
            && !Successor->empty()) {
          auto *Call = dyn_cast<CallInst>(&*Successor->begin());
          if (Call == nullptr
              || Call->getCalledFunction()->getName() != "newpc") {
            WorkList.push_back(Successor);
          }
        }
      }
    }
  }
}

BasicBlock *JumpTargetManager::getBlockAt(uint64_t PC) {
  auto TargetIt = JumpTargets.find(PC);
  assert(TargetIt != JumpTargets.end());
  return TargetIt->second;
}

// TODO: register Reason
BasicBlock *JumpTargetManager::registerJT(uint64_t PC, JTReason Reason) {
  if (!isExecutableAddress(PC) || !isInstructionAligned(PC))
    return nullptr;

  // Do we already have a BasicBlock for this PC?
  BlockMap::iterator TargetIt = JumpTargets.find(PC);
  if (TargetIt != JumpTargets.end()) {
    // Case 1: there's already a BasicBlock for that address, return it
    unvisit(TargetIt->second);
    return TargetIt->second;
  }

  // Did we already meet this PC (i.e. do we know what's the associated
  // instruction)?
  BasicBlock *NewBlock = nullptr;
  InstructionMap::iterator InstrIt = OriginalInstructionAddresses.find(PC);
  if (InstrIt != OriginalInstructionAddresses.end()) {
    // Case 2: the address has already been met, but needs to be promoted to
    //         BasicBlock level.
    registerOriginalBB(PC, 0);

    BasicBlock *ContainingBlock = InstrIt->second->getParent();
    if (InstrIt->second == &*ContainingBlock->begin())
      NewBlock = ContainingBlock;
    else {
      assert(InstrIt->second != nullptr
             && InstrIt->second != ContainingBlock->end());
      NewBlock = ContainingBlock->splitBasicBlock(InstrIt->second);
    }
    unvisit(NewBlock);
  } else {
    // Case 3: the address has never been met, create a temporary one, register
    // it for future exploration and return it
    NewBlock = BasicBlock::Create(Context, "", TheFunction);
    Unexplored.push_back(BlockWithAddress(PC, NewBlock));
  }

  if (NewBlock->getName().empty()) {
    std::stringstream Name;
    Name << "bb.0x" << std::hex << PC;
    NewBlock->setName(Name.str());
  }

  // Create a case for the address associated to the new block
  auto *PCRegType = PCReg->getType();
  auto *SwitchType = cast<IntegerType>(PCRegType->getPointerElementType());
  DispatcherSwitch->addCase(ConstantInt::get(SwitchType, PC), NewBlock);

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
  DispatcherFail = BasicBlock::Create(Context,
                                      "dispatcher.default",
                                      OutputFunction);
  Builder.SetInsertPoint(DispatcherFail);

  Module *TheModule = TheFunction->getParent();
  auto *UnknownPCTy = FunctionType::get(Type::getVoidTy(Context), { }, false);
  Constant *UnknownPC = TheModule->getOrInsertFunction("unknownPC",
                                                       UnknownPCTy);
  Builder.CreateCall(cast<Function>(UnknownPC));
  Builder.CreateUnreachable();

  // Switch on the first argument of the function
  Builder.SetInsertPoint(Entry);
  Value *SwitchOn = Builder.CreateLoad(SwitchOnPtr);
  SwitchInst *Switch = Builder.CreateSwitch(SwitchOn, DispatcherFail);

  Dispatcher = Entry;
  DispatcherSwitch = Switch;
}

// Harvesting proceeds trying to avoid to run expensive analyses if not strictly
// necessary, OSRA in particular. To do this we keep in mind two aspects: do we
// have new basic blocks to visit? If so, we avoid any further anyalysis and
// give back control to the translator. If not, we proceed with other analyses
// until we either find a new basic block to translate. If we can't find a new
// block to translate we proceed as long as we are able to create new edges on
// the CFG (not considering the dispatcher).
void JumpTargetManager::harvest() {
  if (empty()) {
    DBG("verify", if (verifyModule(TheModule, &dbgs())) { abort(); });

    DBG("jtcount", dbg << "Harvesting: SROA, ConstProp, EarlyCSE and SET\n");

    legacy::PassManager PM;
    PM.add(createSROAPass()); // temp
    PM.add(createConstantPropagationPass()); // temp
    PM.add(createEarlyCSEPass());
    PM.add(new SETPass(this, false, &Visited));
    PM.add(new TranslateDirectBranchesPass(this));
    NewBranches = 0;
    PM.run(TheModule);

    DBG("jtcount", dbg << std::dec
                       << Unexplored.size() << " new jump targets and "
                       << NewBranches << " new branches were found\n");
  }

  if (EnableOSRA && empty()) {
    DBG("verify", if (verifyModule(TheModule, &dbgs())) { abort(); });

    do {

      DBG("jtcount",
          dbg << "Harvesting: reset Visited, "
              << (NewBranches > 0 ? "SROA, ConstProp, EarlyCSE, " : "")
              << "SET + OSRA\n");

      // TODO: decide what to do with Visited
      Visited.clear();
      legacy::PassManager PM;
      if (NewBranches > 0) {
        PM.add(createSROAPass()); // temp
        PM.add(createConstantPropagationPass()); // temp
        PM.add(createEarlyCSEPass());
      }
      PM.add(new SETPass(this, true, &Visited));
      PM.add(new TranslateDirectBranchesPass(this));
      NewBranches = 0;
      PM.run(TheModule);

      DBG("jtcount", dbg << std::dec
                         << Unexplored.size() << " new jump targets and "
                         << NewBranches << " new branches were found\n");

    } while (empty() && NewBranches > 0);
  }

  if (empty()) {
    DBG("jtcount", dbg<< "We're done looking for jump targets\n");
  }

}

const JumpTargetManager::BlockWithAddress JumpTargetManager::NoMoreTargets =
  JumpTargetManager::BlockWithAddress(0, nullptr);
