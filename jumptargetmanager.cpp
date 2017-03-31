/// \file jumptargetmanager.cpp
/// \brief This file handles the possible jump targets encountered during
///        translation and the creation and management of the respective
///        BasicBlock.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cassert>
#include <cstdint>
#include <fstream>
#include <queue>
#include <sstream>

// Boost includes
#include <boost/icl/interval_set.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/icl/right_open_interval.hpp>

// LLVM includes
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Endian.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"

// Local includes
#include "datastructures.h"
#include "debug.h"
#include "generatedcodebasicinfo.h"
#include "ir-helpers.h"
#include "jumptargetmanager.h"
#include "revamb.h"
#include "set.h"
#include "simplifycomparisons.h"
#include "subgraph.h"

using namespace llvm;

static bool isSumJump(StoreInst *PCWrite);

char TranslateDirectBranchesPass::ID = 0;

static RegisterPass<TranslateDirectBranchesPass> X("translate-db",
                                                   "Translate Direct Branches"
                                                   " Pass",
                                                   false,
                                                   false);

void TranslateDirectBranchesPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addUsedIfAvailable<SETPass>();
  AU.setPreservesAll();
}

/// \brief Purges everything is after a call to exitTB (except the call itself)
static void exitTBCleanup(Instruction *ExitTBCall) {
  BasicBlock *BB = ExitTBCall->getParent();

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

  LLVMContext &Context = getContext(&F);
  Value *PCReg = JTM->pcReg();
  auto *RegType = cast<IntegerType>(PCReg->getType()->getPointerElementType());
  auto C = [RegType] (uint64_t A) { return ConstantInt::get(RegType, A); };
  BasicBlock *AnyPC = JTM->anyPC();
  BasicBlock *UnexpectedPC = JTM->unexpectedPC();
  // TODO: enforce CFG

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

    BasicBlock *FailBB = Approximate ? AnyPC : UnexpectedPC;
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
  auto &Context = F.getParent()->getContext();

  Function *ExitTB = JTM->exitTB();
  auto ExitTBIt = ExitTB->use_begin();
  while (ExitTBIt != ExitTB->use_end()) {
    // Take note of the use and increment the iterator immediately: this allows
    // us to erase the call to exit_tb without unexpected behaviors
    Use &ExitTBUse = *ExitTBIt++;
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
                       JTM->anyPC());

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
          uint64_t PC = getLimitedValue(Marker->getArgOperand(0));
          uint64_t Size = getLimitedValue(Marker->getArgOperand(1));
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

Optional<uint64_t>
JumpTargetManager::readRawValue(uint64_t Address,
                                unsigned Size,
                                Endianess ReadEndianess) const {
  bool IsLittleEndian;
  if (ReadEndianess == OriginalEndianess) {
    IsLittleEndian = Binary.architecture().isLittleEndian();
  } else if (ReadEndianess == DestinationEndianess) {
    IsLittleEndian = TheModule.getDataLayout().isLittleEndian();
  } else {
    abort();
  }

  for (auto &Segment : Binary.segments()) {
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
      switch (Size) {
      case 1:
        return read<uint8_t, endianness::little, 1>(Start);
      case 2:
        if (IsLittleEndian)
          return read<uint16_t, endianness::little, 1>(Start);
        else
          return read<uint16_t, endianness::big, 1>(Start);
      case 4:
        if (IsLittleEndian)
          return read<uint32_t, endianness::little, 1>(Start);
        else
          return read<uint32_t, endianness::big, 1>(Start);
      case 8:
        if (IsLittleEndian)
          return read<uint64_t, endianness::little, 1>(Start);
        else
          return read<uint64_t, endianness::big, 1>(Start);
      default:
        assert(false && "Unexpected read size");
      }
    }
  }

  return Optional<uint64_t>();
}

Constant *JumpTargetManager::readConstantPointer(Constant *Address,
                                                 Type *PointerTy,
                                                 Endianess ReadEndianess) {
  auto *Value = readConstantInt(Address,
                                Binary.architecture().pointerSize() / 8,
                                ReadEndianess);
  if (Value != nullptr) {
    return ConstantExpr::getIntToPtr(Value, PointerTy);
  } else {
    return nullptr;
  }
}

ConstantInt *JumpTargetManager::readConstantInt(Constant *ConstantAddress,
                                                unsigned Size,
                                                Endianess ReadEndianess) {
  const DataLayout &DL = TheModule.getDataLayout();

  if (ConstantAddress->getType()->isPointerTy()) {
    using CE = ConstantExpr;
    auto IntPtrTy = Type::getIntNTy(Context,
                                    Binary.architecture().pointerSize());
    ConstantAddress = CE::getPtrToInt(ConstantAddress, IntPtrTy);
  }

  uint64_t Address = getZExtValue(ConstantAddress, DL);
  UnusedCodePointers.erase(Address);
  registerReadRange(Address, Size);

  auto Result = readRawValue(Address, Size, ReadEndianess);

  if (Result.hasValue())
    return ConstantInt::get(IntegerType::get(Context, Size * 8),
                            Result.getValue());
  else
    return nullptr;
}

template<typename T>
static cl::opt<T> *getOption(StringMap<cl::Option *>& Options,
                             const char *Name) {
  return static_cast<cl::opt<T> *>(Options[Name]);
}

JumpTargetManager::JumpTargetManager(Function *TheFunction,
                                     Value *PCReg,
                                     const BinaryFile &Binary,
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
  Binary(Binary),
  EnableOSRA(EnableOSRA),
  NoReturn(Binary.architecture()),
  CurrentCFGForm(UnknownFormCFG) {
  FunctionType *ExitTBTy = FunctionType::get(Type::getVoidTy(Context),
                                             { Type::getInt32Ty(Context) },
                                             false);
  ExitTB = cast<Function>(TheModule.getOrInsertFunction("exitTB", ExitTBTy));
  createDispatcher(TheFunction, PCReg, true);

  for (auto &Segment : Binary.segments())
    Segment.insertExecutableRanges(std::back_inserter(ExecutableRanges));

  initializeSymbolMap();

  // Configure GlobalValueNumbering
  StringMap<cl::Option *>& Options(cl::getRegisteredOptions());
  getOption<bool>(Options, "enable-load-pre")->setInitialValue(false);
  getOption<unsigned>(Options, "memdep-block-scan-limit")->setInitialValue(100);
  // getOption<bool>(Options, "enable-pre")->setInitialValue(false);
  // getOption<uint32_t>(Options, "max-recurse-depth")->setInitialValue(10);
}

void JumpTargetManager::initializeSymbolMap() {
  // Collect how many times each name is used
  std::map<std::string, unsigned> SeenCount;
  for (const SymbolInfo &Symbol : Binary.symbols())
    SeenCount[std::string(Symbol.Name)]++;

  for (const SymbolInfo &Symbol : Binary.symbols()) {
    // Discard symbols pointing to 0, with zero-sized names or present multiple
    // times. Note that we keep zero-size symbols.
    if (Symbol.Address == 0
        || Symbol.Name.size() == 0
        || SeenCount[std::string(Symbol.Name)] > 1)
      continue;

    // Associate to this interval the symbol
    unsigned Size = std::max(1UL, Symbol.Size);
    auto NewInterval = interval::right_open(Symbol.Address,
                                            Symbol.Address + Size);
    SymbolMap += make_pair(NewInterval, SymbolInfoSet { &Symbol });
  }
}

// TODO: move this in BinaryFile?
std::string JumpTargetManager::nameForAddress(uint64_t Address) const {
  std::stringstream Result;

  // Take the interval greater than [Address, Address + 1[
  auto It = SymbolMap.upper_bound(interval::right_open(Address, Address + 1));
  if (It != SymbolMap.begin()) {
    // Go back one position
    It--;

    // In case we have multiple matching symbols, take the closest one
    const SymbolInfoSet &Matching = It->second;
    auto MaxIt = std::max_element(Matching.begin(), Matching.end());
    const SymbolInfo *const BestMatch = *MaxIt;

    // Use the symbol name
    Result << BestMatch->Name.str();

    // And, if necessary, an offset
    if (Address != BestMatch->Address)
      Result << ".0x" << std::hex << (Address - BestMatch->Address);
  } else {
    // We don't have a symbol to use, just return the address
    Result << "0x" << std::hex << Address;
  }

  return Result.str();
}

void JumpTargetManager::harvestGlobalData() {
  // Register landing pads, if available
  // TODO: should register them in UnusedCodePointers?
  for (uint64_t LandingPad : Binary.landingPads())
    registerJT(LandingPad, GlobalData);

  for (auto& Segment : Binary.segments()) {
    auto *Data = cast<ConstantDataArray>(Segment.Variable->getInitializer());
    uint64_t StartVirtualAddress = Segment.StartVirtualAddress;
    const unsigned char *DataStart = Data->getRawDataValues().bytes_begin();
    const unsigned char *DataEnd = Data->getRawDataValues().bytes_end();

    using endianness = support::endianness;
    if (Binary.architecture().pointerSize() == 64) {
      if (Binary.architecture().isLittleEndian())
        findCodePointers<uint64_t, endianness::little>(StartVirtualAddress,
                                                       DataStart,
                                                       DataEnd);
      else
        findCodePointers<uint64_t, endianness::big>(StartVirtualAddress,
                                                    DataStart,
                                                    DataEnd);
    } else if (Binary.architecture().pointerSize() == 32) {
      if (Binary.architecture().isLittleEndian())
        findCodePointers<uint32_t, endianness::little>(StartVirtualAddress,
                                                       DataStart,
                                                       DataEnd);
      else
        findCodePointers<uint32_t, endianness::big>(StartVirtualAddress,
                                                    DataStart,
                                                    DataEnd);
    }
  }

  DBG("jtcount", dbg
      << "JumpTargets found in global data: " << std::dec
      << Unexplored.size() << "\n");
}

template<typename value_type, unsigned endian>
void JumpTargetManager::findCodePointers(uint64_t StartVirtualAddress,
                                         const unsigned char *Start,
                                         const unsigned char *End) {
  using support::endian::read;
  using support::endianness;
  for (auto Pos = Start; Pos < End - sizeof(value_type); Pos++) {
    uint64_t Value = read<value_type,
                          static_cast<endianness>(endian),
                          1>(Pos);
    BasicBlock *Result = registerJT(Value, GlobalData);

    if (Result != nullptr)
      UnusedCodePointers.insert(StartVirtualAddress + (Pos - Start));
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
    BasicBlock *BB = JTIt->second.head();
    assert(!BB->empty());
    ShouldContinue = false;
    return BB;
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

  visitSuccessors(Start,
                  make_blacklist(*this),
                  [this,&Result] (BasicBlockRange Range) {
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


// TODO: this is outdated and we should drop it, we now have OSRA and friends
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

  uint64_t PC = getLimitedValue(NewPCCall->getArgOperand(0));
  uint64_t Size = getLimitedValue(NewPCCall->getArgOperand(1));
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
          uint64_t PC = getLimitedValue(Call->getArgOperand(0));

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
          NextPC = PC + getLimitedValue(Call->getArgOperand(1));
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

  // Purge all the partial translations we know might be wrong
  for (BasicBlock *BB : ToPurge)
    purgeTranslation(BB);
  ToPurge.clear();

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
  return TargetIt->second.head();
}

void JumpTargetManager::purgeTranslation(BasicBlock *Start) {
  OnceQueue<BasicBlock *> Queue;
  Queue.insert(Start);

  // Collect all the descendats, except if we meet a jump target
  while (!Queue.empty()) {
    BasicBlock *BB = Queue.pop();
    for (BasicBlock *Successor : successors(BB)) {
      if (isTranslatedBB(Successor)
          && !isJumpTarget(Successor)
          && !hasPredecessor(Successor, Dispatcher)) {
        Queue.insert(Successor);
      }
    }
  }

  // Erase all the visited basic blocks
  std::set<BasicBlock *> Visited = Queue.visited();

  // Build a subgraph, so that we can visit it in post order, and purge the
  // content of each basic block
  SubGraph<BasicBlock *> TranslatedBBs(Start, Visited);
  for (auto *Node : post_order(TranslatedBBs)) {
    BasicBlock *BB = Node->get();
    while (!BB->empty())
      eraseInstruction(&*(--BB->end()));
  }

  // Remove Start, since we want to keep it (even if empty)
  Visited.erase(Start);

  for (BasicBlock *BB : Visited) {
    // We might have some predecessorless basic blocks jumping to us, purge them
    // TODO: why this?
    while (pred_begin(BB) != pred_end(BB)) {
      BasicBlock *Predecessor = *pred_begin(BB);
      assert(pred_empty(Predecessor));
      Predecessor->eraseFromParent();
    }

    assert(BB->use_empty());
    BB->eraseFromParent();
  }
}

// TODO: register Reason
BasicBlock *JumpTargetManager::registerJT(uint64_t PC, JTReason Reason) {
  if (!isExecutableAddress(PC) || !isInstructionAligned(PC))
    return nullptr;

  // Do we already have a BasicBlock for this PC?
  BlockMap::iterator TargetIt = JumpTargets.find(PC);
  if (TargetIt != JumpTargets.end()) {
    // Case 1: there's already a BasicBlock for that address, return it
    BasicBlock *BB = TargetIt->second.head();
    TargetIt->second.setReason(Reason);
    unvisit(BB);
    return BB;
  }

  // Did we already meet this PC (i.e. do we know what's the associated
  // instruction)?
  BasicBlock *NewBlock = nullptr;
  InstructionMap::iterator InstrIt = OriginalInstructionAddresses.find(PC);
  if (InstrIt != OriginalInstructionAddresses.end()) {
    // Case 2: the address has already been met, but needs to be promoted to
    //         BasicBlock level.
    Instruction *I = InstrIt->second;
    BasicBlock *ContainingBlock = I->getParent();
    if (isFirst(I)) {
      NewBlock = ContainingBlock;
    } else {
      assert(I != nullptr && I != ContainingBlock->end());
      NewBlock = ContainingBlock->splitBasicBlock(I);
    }

    // Register the basic block and all of its descendants to be purged so that
    // we can retranslate this PC
    // TODO: this might create a problem if QEMU generates control flow that
    //       crosses an instruction boundary
    ToPurge.insert(NewBlock);

    unvisit(NewBlock);
  } else {
    // Case 3: the address has never been met, create a temporary one, register
    // it for future exploration and return it
    NewBlock = BasicBlock::Create(Context, "", TheFunction);
  }

  Unexplored.push_back(BlockWithAddress(PC, NewBlock));

  if (NewBlock->getName().empty()) {
    std::stringstream Name;
    Name << "bb." << nameForAddress(PC);
    NewBlock->setName(Name.str());
  }

  // Create a case for the address associated to the new block
  auto *PCRegType = PCReg->getType();
  auto *SwitchType = cast<IntegerType>(PCRegType->getPointerElementType());
  DispatcherSwitch->addCase(ConstantInt::get(SwitchType, PC), NewBlock);

  // Associate the PC with the chosen basic block
  JumpTargets[PC] = JumpTarget(NewBlock, Reason);
  return NewBlock;
}

void JumpTargetManager::registerReadRange(uint64_t Address, uint64_t Size) {
  using interval = boost::icl::interval<uint64_t>;
  ReadIntervalSet += interval::right_open(Address, Address + Size);
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
  // The switch is the terminator of the dispatcher basic block
  QuickMetadata QMD(Context);
  Switch->setMetadata("revamb.block.type", QMD.tuple(DispatcherBlock));

  Dispatcher = Entry;
  DispatcherSwitch = Switch;
  NoReturn.setDispatcher(Dispatcher);

  // Create basic blocks to handle jumps to any PC and to a PC we didn't expect
  AnyPC = BasicBlock::Create(Context, "anypc", OutputFunction);
  UnexpectedPC = BasicBlock::Create(Context, "unexpectedpc", OutputFunction);

  setCFGForm(SemanticPreservingCFG);
}

static void purge(BasicBlock *BB) {
  // Allow up to a single instruction in the basic block
  if (!BB->empty())
    BB->begin()->eraseFromParent();
  assert(BB->empty());
}

void JumpTargetManager::setCFGForm(CFGForm NewForm) {
  assert(CurrentCFGForm != NewForm);
  assert(NewForm != UnknownFormCFG);

  CFGForm OldForm = CurrentCFGForm;
  CurrentCFGForm = NewForm;

  switch (NewForm) {
  case SemanticPreservingCFG:
    purge(AnyPC);
    BranchInst::Create(dispatcher(), AnyPC);
    // TODO: Here we should have an hard fail, since it's the situation in
    //       which we expected to know where execution could go but we made a
    //       mistake.
    purge(UnexpectedPC);
    BranchInst::Create(dispatcher(), UnexpectedPC);
    break;

  case RecoveredOnlyCFG:
  case NoFunctionCallsCFG:
    purge(AnyPC);
    new UnreachableInst(Context, AnyPC);
    purge(UnexpectedPC);
    new UnreachableInst(Context, UnexpectedPC);
    break;

  default:
    assert(false && "Not implemented yet");
    break;
  }

  QuickMetadata QMD(Context);
  AnyPC->getTerminator()->setMetadata("revamb.block.type",
                                      QMD.tuple(AnyPCBlock));
  UnexpectedPC->getTerminator()->setMetadata("revamb.block.type",
                                             QMD.tuple(UnexpectedPCBlock));

  // If we're entering or leaving the NoFunctionCallsCFG form, update all the
  // branch instruction forming a function call
  if (NewForm == NoFunctionCallsCFG || OldForm == NoFunctionCallsCFG) {
    if (auto *FunctionCall = TheModule.getFunction("function_call")) {
      for (User *U : FunctionCall->users()) {
        auto *Call = cast<CallInst>(U);
        auto *Terminator = cast<TerminatorInst>(Call->getNextNode());
        assert(Terminator->getNumSuccessors() == 1);

        // Get the correct argument, the first is the callee, the second the
        // return basic block
        Value *Op = Call->getArgOperand(NewForm == NoFunctionCallsCFG ? 1 : 0);
        BasicBlock *NewSuccessor = cast<BlockAddress>(Op)->getBasicBlock();
        Terminator->setSuccessor(0, NewSuccessor);
      }
    }
  }

  rebuildDispatcher();
}

void JumpTargetManager::rebuildDispatcher() {
  // Remove all cases
  unsigned NumCases = DispatcherSwitch->getNumCases();
  while (NumCases --> 0)
    DispatcherSwitch->removeCase(DispatcherSwitch->case_begin());

  auto *PCRegType = PCReg->getType()->getPointerElementType();
  auto *SwitchType = cast<IntegerType>(PCRegType);

  // Add all the jump targets if we're using the SemanticPreservingCFG, or
  // only those with no predecessors otherwise
  for (auto &P : JumpTargets) {
    uint64_t PC = P.first;
    BasicBlock *BB = P.second.head();
    if (CurrentCFGForm == SemanticPreservingCFG || !hasPredecessors(BB))
      DispatcherSwitch->addCase(ConstantInt::get(SwitchType, PC), BB);

  }
}

bool JumpTargetManager::hasPredecessors(BasicBlock *BB) const {
  for (BasicBlock *Pred : predecessors(BB))
    if (isTranslatedBB(Pred))
      return true;
  return false;
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
    // TODO: move me to a commit function
    // Update the third argument of newpc calls (isJT, i.e., is this instruction
    // a jump target?)
    IRBuilder<> Builder(Context);
    Function *NewPCFunction = TheModule.getFunction("newpc");
    if (NewPCFunction != nullptr) {
      for (User *U : NewPCFunction->users()) {
        auto *Call = cast<CallInst>(U);
        if (Call->getParent() != nullptr) {
          // Report the instruction on the coverage CSV
          using CI = ConstantInt;
          uint64_t PC = (cast<CI>(Call->getArgOperand(0)))->getLimitedValue();

          bool IsJT = isJumpTarget(PC);
          Call->setArgOperand(2, Builder.getInt32(static_cast<uint32_t>(IsJT)));
        }
      }
    }

    DBG("verify", if (verifyModule(TheModule, &dbgs())) { abort(); });

    DBG("jtcount", dbg << "Harvesting: SROA, ConstProp, EarlyCSE and SET\n");

    legacy::PassManager OptimizingPM;
    OptimizingPM.add(createSROAPass());
    OptimizingPM.add(createConstantPropagationPass());
    OptimizingPM.add(createEarlyCSEPass());
    OptimizingPM.run(TheModule);

    // To improve the quality of our analysis, keep in the CFG only the edges we
    // where able to recover (e.g., no jumps to the dispatcher)
    setCFGForm(RecoveredOnlyCFG);

    NewBranches = 0;
    legacy::PassManager AnalysisPM;
    AnalysisPM.add(new SETPass(this, false, &Visited));
    AnalysisPM.add(new TranslateDirectBranchesPass(this));
    AnalysisPM.run(TheModule);

    // Restore the CFG
    setCFGForm(SemanticPreservingCFG);

    DBG("jtcount", dbg << std::dec
                       << Unexplored.size() << " new jump targets and "
                       << NewBranches << " new branches were found\n");
  }

  if (EnableOSRA && empty()) {
    DBG("verify", if (verifyModule(TheModule, &dbgs())) { abort(); });

    NoReturn.registerSyscalls(TheFunction);

    do {

      DBG("jtcount",
          dbg << "Harvesting: reset Visited, "
              << (NewBranches > 0 ? "SROA, ConstProp, EarlyCSE, " : "")
              << "SET + OSRA\n");

      // TODO: decide what to do with Visited
      Visited.clear();
      legacy::PassManager PM;
      if (NewBranches > 0) {
        legacy::PassManager OptimizingPM;
        OptimizingPM.add(createSROAPass());
        OptimizingPM.add(createConstantPropagationPass());
        OptimizingPM.add(createEarlyCSEPass());
        OptimizingPM.run(TheModule);
      }

      setCFGForm(RecoveredOnlyCFG);

      NewBranches = 0;
      legacy::PassManager AnalysisPM;
      AnalysisPM.add(new SETPass(this, true, &Visited));
      AnalysisPM.add(new TranslateDirectBranchesPass(this));
      AnalysisPM.run(TheModule);

      // Restore the CFG
      setCFGForm(SemanticPreservingCFG);

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
