/// \file jumptargetmanager.cpp
/// \brief This file handles the possible jump targets encountered during
///        translation and the creation and management of the respective
///        BasicBlock.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include "revng/Support/Assert.h"
#include <cstdint>
#include <fstream>
#include <queue>
#include <sstream>

// Boost includes
#include <boost/icl/interval_set.hpp>
#include <boost/icl/right_open_interval.hpp>
#include <boost/type_traits/is_same.hpp>

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

// Local libraries includes
#include "revng/ADT/Queue.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/BasicAnalyses/ReachingDefinitionsPass.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/revng.h"

// Local includes
#include "JumpTargetManager.h"
#include "SET.h"
#include "SimplifyComparisonsPass.h"
#include "SubGraph.h"

using namespace llvm;

namespace {

Logger<> JTCountLog("jtcount");

cl::opt<bool> NoOSRA("no-osra", cl::desc(" OSRA"), cl::cat(MainCategory));
cl::alias A1("O",
             cl::desc("Alias for -no-osra"),
             cl::aliasopt(NoOSRA),
             cl::cat(MainCategory));

RegisterPass<TranslateDirectBranchesPass> X("translate-db",
                                            "Translate Direct Branches"
                                            " Pass",
                                            false,
                                            false);

// TODO: this is kind of an abuse
Logger<> Verify("verify");
Logger<> RegisterJTLog("registerjt");

} // namespace

char TranslateDirectBranchesPass::ID = 0;

static bool isSumJump(StoreInst *PCWrite);

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
  auto C = [RegType](uint64_t A) { return ConstantInt::get(RegType, A); };
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

    revng_assert(CallExitTB != nullptr);
    revng_assert(PCWrite->getParent()->getParent() == &F);
    revng_assert(JTM->isPCReg(PCWrite->getPointerOperand()));
    revng_assert(Destinations.size() != 0);

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

    // Move all the markers right before the branch instruction
    Instruction *Last = BB->getTerminator();
    auto It = CallExitTB->getIterator();
    while (isMarker(&*It)) {
      // Get the marker instructions
      Instruction *I = &*It;

      // Move the iterator back
      It--;

      // Move the last moved instruction (initially the terminator)
      I->moveBefore(Last);

      Last = I;
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
          if (NextPC != 0 && not NoOSRA && isSumJump(PCWrite))
            JTM->registerJT(NextPC, JTReason::SumJump);

          auto *Address = dyn_cast<ConstantInt>(PCWrite->getValueOperand());
          if (Address != nullptr) {
            // Compute the actual PC and get the associated BasicBlock
            uint64_t TargetPC = Address->getSExtValue();
            auto *TargetBlock = JTM->registerJT(TargetPC, JTReason::DirectJump);

            // Remove unreachable right after the exit_tb
            BasicBlock::iterator CallIt(Call);
            BasicBlock::iterator BlockEnd = Call->getParent()->end();
            CallIt++;
            revng_assert(CallIt != BlockEnd && isa<UnreachableInst>(&*CallIt));
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
              CallInst::Create(F.getParent()->getFunction("abort"), {}, Call);
              new UnreachableInst(Context, Call);
            }
            Call->eraseFromParent();
          }
        }
      } else {
        revng_unreachable("Unexpected instruction using the PC");
      }
    } else {
      revng_unreachable("Unhandled usage of the PC");
    }
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

  BasicBlock::reverse_iterator It(++Call->getReverseIterator());
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

  // Get the fallthrough basic block and emit a conditional branch, if not
  // possible simply jump to anyPC
  BasicBlock *NextPCBB = JTM->registerJT(NextPC, JTReason::PostHelper);
  if (NextPCBB != nullptr) {
    Builder.CreateCondBr(Builder.CreateICmpEQ(Builder.CreateLoad(PCReg),
                                              NextPCConst),
                         NextPCBB,
                         JTM->anyPC());
  } else {
    Builder.CreateBr(JTM->anyPC());
  }

  return true;
}

bool TranslateDirectBranchesPass::runOnFunction(Function &F) {
  pinConstantStore(F);
  pinJTs(F);
  return true;
}

uint64_t TranslateDirectBranchesPass::getNextPC(Instruction *TheInstruction) {
  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  BasicBlock *Block = TheInstruction->getParent();
  BasicBlock::reverse_iterator It(++TheInstruction->getReverseIterator());

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
          revng_assert(Size != 0);
          return PC + Size;
        }
      }
    }

    auto *Node = DT.getNode(Block);
    revng_assert(Node != nullptr,
                 "BasicBlock not in the dominator tree, is it reachable?");

    Block = Node->getIDom()->getBlock();
    It = Block->rbegin();
  }

  revng_unreachable("Can't find the PC marker");
}

Constant *JumpTargetManager::readConstantPointer(Constant *Address,
                                                 Type *PointerTy,
                                                 BinaryFile::Endianess E) {
  Constant *ConstInt = readConstantInt(Address,
                                       Binary.architecture().pointerSize() / 8,
                                       E);
  if (ConstInt != nullptr) {
    return Constant::getIntegerValue(PointerTy, ConstInt->getUniqueInteger());
  } else {
    return nullptr;
  }
}

ConstantInt *JumpTargetManager::readConstantInt(Constant *ConstantAddress,
                                                unsigned Size,
                                                BinaryFile::Endianess E) {
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

  auto Result = Binary.readRawValue(Address, Size, E);

  if (Result.hasValue())
    return ConstantInt::get(IntegerType::get(Context, Size * 8),
                            Result.getValue());
  else
    return nullptr;
}

template<typename T>
static cl::opt<T> *
getOption(StringMap<cl::Option *> &Options, const char *Name) {
  return static_cast<cl::opt<T> *>(Options[Name]);
}

JumpTargetManager::JumpTargetManager(Function *TheFunction,
                                     Value *PCReg,
                                     const BinaryFile &Binary) :
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
  NoReturn(Binary.architecture()),
  CurrentCFGForm(CFGForm::UnknownFormCFG) {
  FunctionType *ExitTBTy = FunctionType::get(Type::getVoidTy(Context),
                                             { Type::getInt32Ty(Context) },
                                             false);
  ExitTB = cast<Function>(TheModule.getOrInsertFunction("exitTB", ExitTBTy));
  createDispatcher(TheFunction, PCReg);

  for (auto &Segment : Binary.segments())
    Segment.insertExecutableRanges(std::back_inserter(ExecutableRanges));

  // Configure GlobalValueNumbering
  StringMap<cl::Option *> &Options(cl::getRegisteredOptions());
  getOption<bool>(Options, "enable-load-pre")->setInitialValue(false);
  getOption<unsigned>(Options, "memdep-block-scan-limit")->setInitialValue(100);
  // getOption<bool>(Options, "enable-pre")->setInitialValue(false);
  // getOption<uint32_t>(Options, "max-recurse-depth")->setInitialValue(10);
}

static bool isBetterThan(const Label *NewCandidate, const Label *OldCandidate) {
  if (OldCandidate == nullptr)
    return true;

  if (NewCandidate->address() > OldCandidate->address())
    return true;

  if (NewCandidate->address() == OldCandidate->address()) {
    StringRef OldName = OldCandidate->symbolName();
    if (OldName.size() == 0)
      return true;
  }

  return false;
}

// TODO: move this in BinaryFile?
std::string
JumpTargetManager::nameForAddress(uint64_t Address, uint64_t Size) const {
  std::stringstream Result;
  const auto &SymbolMap = Binary.labels();

  auto It = SymbolMap.find(interval::right_open(Address, Address + Size));
  if (It != SymbolMap.end()) {
    // We have to look for (in order):
    //
    // * Exact match
    // * Contained (non 0-sized)
    // * Contained (0-sized)
    const Label *ExactMatch = nullptr;
    const Label *ContainedNonZeroSized = nullptr;
    const Label *ContainedZeroSized = nullptr;

    for (const Label *L : It->second) {
      // Consider symbols only
      if (not L->isSymbol())
        continue;

      if (L->matches(Address, Size)) {

        // It's an exact match
        ExactMatch = L;
        break;

      } else if (not L->isSizeVirtual() and L->contains(Address, Size)) {

        // It's contained in a not 0-sized symbol
        if (isBetterThan(L, ContainedNonZeroSized))
          ContainedNonZeroSized = L;

      } else if (L->isSizeVirtual() and L->contains(Address, 0)) {

        // It's contained in a 0-sized symbol
        if (isBetterThan(L, ContainedZeroSized))
          ContainedZeroSized = L;
      }
    }

    const Label *Chosen = nullptr;
    if (ExactMatch != nullptr)
      Chosen = ExactMatch;
    else if (ContainedNonZeroSized != nullptr)
      Chosen = ContainedNonZeroSized;
    else if (ContainedZeroSized != nullptr)
      Chosen = ContainedZeroSized;

    if (Chosen != nullptr and Chosen->symbolName().size() != 0) {
      // Use the symbol name
      Result << Chosen->symbolName().str();

      // And, if necessary, an offset
      if (Address != Chosen->address())
        Result << ".0x" << std::hex << (Address - Chosen->address());

      return Result.str();
    }
  }

  // We don't have a symbol to use, just return the address
  Result << "0x" << std::hex << Address;
  return Result.str();
}

void JumpTargetManager::harvestGlobalData() {
  // Register symbols
  for (auto &P : Binary.labels())
    for (const Label *L : P.second)
      if (L->isSymbol() and L->isCode())
        registerJT(L->address(), JTReason::FunctionSymbol);

  // Register landing pads, if available
  // TODO: should register them in UnusedCodePointers?
  for (uint64_t LandingPad : Binary.landingPads())
    registerJT(LandingPad, JTReason::GlobalData);

  for (uint64_t CodePointer : Binary.codePointers())
    registerJT(CodePointer, JTReason::GlobalData);

  for (auto &Segment : Binary.segments()) {
    const Constant *Initializer = Segment.Variable->getInitializer();
    if (isa<ConstantAggregateZero>(Initializer))
      continue;

    auto *Data = cast<ConstantDataArray>(Initializer);
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

  revng_log(JTCountLog,
            "JumpTargets found in global data: " << std::dec
                                                 << Unexplored.size());
}

template<typename value_type, unsigned endian>
void JumpTargetManager::findCodePointers(uint64_t StartVirtualAddress,
                                         const unsigned char *Start,
                                         const unsigned char *End) {
  using support::endianness;
  using support::endian::read;
  for (auto Pos = Start; Pos < End - sizeof(value_type); Pos++) {
    uint64_t Value = read<value_type, static_cast<endianness>(endian), 1>(Pos);
    BasicBlock *Result = registerJT(Value, JTReason::GlobalData);

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
BasicBlock *JumpTargetManager::newPC(uint64_t PC, bool &ShouldContinue) {
  // Did we already meet this PC?
  auto JTIt = JumpTargets.find(PC);
  if (JTIt != JumpTargets.end()) {
    // If it was planned to explore it in the future, just to do it now
    for (auto UnexploredIt = Unexplored.begin();
         UnexploredIt != Unexplored.end();
         UnexploredIt++) {

      if (UnexploredIt->first == PC) {
        BasicBlock *Result = UnexploredIt->second;

        // Check if we already have a translation for that
        ShouldContinue = Result->empty();
        if (ShouldContinue) {
          // We don't, OK let's explore it next
          Unexplored.erase(UnexploredIt);
        } else {
          // We do, it will be purged at the next `peek`
          revng_assert(ToPurge.count(Result) != 0);
        }

        return Result;
      }
    }

    // It wasn't planned to visit it, so we've already been there, just jump
    // there
    BasicBlock *BB = JTIt->second.head();
    revng_assert(!BB->empty());
    ShouldContinue = false;
    return BB;
  }

  // Check if we already translated this PC even if it's not associated to a
  // basic block (i.e., we have to split its basic block). This typically
  // happens with variable-length instruction encodings.
  if (OriginalInstructionAddresses.count(PC) != 0) {
    ShouldContinue = false;
    return registerJT(PC, JTReason::AmbigousInstruction);
  }

  // We don't know anything about this PC
  return nullptr;
}

/// Save the PC-Instruction association for future use (jump target)
void JumpTargetManager::registerInstruction(uint64_t PC,
                                            Instruction *Instruction) {
  // Never save twice a PC
  revng_assert(!OriginalInstructionAddresses.count(PC));
  OriginalInstructionAddresses[PC] = Instruction;
}

CallInst *JumpTargetManager::findNextExitTB(Instruction *Start) {

  struct Visitor
    : public BFSVisitorBase<true, Visitor, SmallVector<BasicBlock *, 4>> {
  public:
    using SuccessorsType = SmallVector<BasicBlock *, 4>;

  public:
    CallInst *Result;
    Function *ExitTB;
    JumpTargetManager *JTM;

  public:
    Visitor(Function *ExitTB, JumpTargetManager *JTM) :
      Result(nullptr),
      ExitTB(ExitTB),
      JTM(JTM) {}

  public:
    VisitAction visit(BasicBlockRange Range) {
      for (Instruction &I : Range) {
        if (auto *Call = dyn_cast<CallInst>(&I)) {
          revng_assert(!(Call->getCalledFunction()->getName() == "newpc"));
          if (Call->getCalledFunction() == ExitTB) {
            revng_assert(Result == nullptr);
            Result = Call;
            return ExhaustQueueAndStop;
          }
        }
      }

      return Continue;
    }

    SuccessorsType successors(BasicBlock *BB) {
      SuccessorsType Successors;
      for (BasicBlock *Successor : make_range(succ_begin(BB), succ_end(BB)))
        if (JTM->isTranslatedBB(Successor))
          Successors.push_back(Successor);
      return Successors;
    }
  };

  Visitor V(ExitTB, this);
  V.run(Start);

  return V.Result;
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
///     addne pc, pc, \\curbit, lsl #2
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
        for (auto &Operand : BinOp->operands())
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
    WorkList.push(++TheInstruction->getReverseIterator());

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
        revng_assert(!(NewPCCall == nullptr && Predecessor == Dispatcher));
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
  revng_assert(Size != 0);
  return { PC, Size };
}

void JumpTargetManager::handleSumJump(Instruction *SumJump) {
  // Take the next PC
  uint64_t NextPC = getNextPC(SumJump);
  revng_assert(NextPC != 0);
  BasicBlock *BB = registerJT(NextPC, JTReason::SumJump);
  revng_assert(BB && !BB->empty());

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
          BB = registerJT(PC, JTReason::SumJump);

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
    DL(Dispatcher->getParent()->getParent()->getParent()->getDataLayout()) {}

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
    Use &ExitTBUse = *I++;
    if (auto *Call = dyn_cast<CallInst>(ExitTBUse.getUser())) {
      if (Call->getCalledFunction() == ExitTB) {

        // Look for the last write to the PC
        StoreInst *PCWrite = getPrevPCWrite(Call);
        if (PCWrite != nullptr) {
          revng_assert(!isa<ConstantInt>(PCWrite->getValueOperand()),
                       "Direct jumps should not be handled here");
        }

        if (PCWrite != nullptr && not NoOSRA && isSumJump(PCWrite))
          handleSumJump(PCWrite);

        if (getLimitedValue(Call->getArgOperand(0)) == 0) {
          exitTBCleanup(Call);
          BranchInst::Create(Dispatcher, Call);
        }

        Call->eraseFromParent();
      }
    }
  }

  revng_assert(ExitTB->use_empty());
  ExitTB->eraseFromParent();
  ExitTB = nullptr;
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
        if (Visited.find(Successor) != Visited.end() && !Successor->empty()) {
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
  revng_assert(TargetIt != JumpTargets.end());
  return TargetIt->second.head();
}

void JumpTargetManager::purgeTranslation(BasicBlock *Start) {
  OnceQueue<BasicBlock *> Queue;
  Queue.insert(Start);

  // Collect all the descendats, except if we meet a jump target
  while (!Queue.empty()) {
    BasicBlock *BB = Queue.pop();
    for (BasicBlock *Successor : successors(BB)) {
      if (isTranslatedBB(Successor) && !isJumpTarget(Successor)
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
      revng_assert(pred_empty(Predecessor));
      Predecessor->eraseFromParent();
    }

    revng_assert(BB->use_empty());
    BB->eraseFromParent();
  }
}

// TODO: register Reason
BasicBlock *
JumpTargetManager::registerJT(uint64_t PC, JTReason::Values Reason) {
  if (!isExecutableAddress(PC) || !isInstructionAligned(PC))
    return nullptr;

  revng_log(RegisterJTLog,
            "Registering bb." << nameForAddress(PC) << " for "
                              << JTReason::getName(Reason));

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
      revng_assert(I != nullptr && I->getIterator() != ContainingBlock->end());
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

  std::stringstream Name;
  Name << "bb." << nameForAddress(PC);
  NewBlock->setName(Name.str());

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
                                         Value *SwitchOnPtr) {
  IRBuilder<> Builder(Context);
  QuickMetadata QMD(Context);

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
  auto *UnknownPCTy = FunctionType::get(Type::getVoidTy(Context), {}, false);
  Constant *UnknownPC = TheModule->getOrInsertFunction("unknownPC",
                                                       UnknownPCTy);
  Builder.CreateCall(cast<Function>(UnknownPC));
  auto *FailUnreachable = Builder.CreateUnreachable();
  FailUnreachable->setMetadata("revamb.block.type",
                               QMD.tuple((uint32_t) DispatcherFailure));

  // Switch on the first argument of the function
  Builder.SetInsertPoint(Entry);
  Value *SwitchOn = Builder.CreateLoad(SwitchOnPtr);
  SwitchInst *Switch = Builder.CreateSwitch(SwitchOn, DispatcherFail);
  // The switch is the terminator of the dispatcher basic block
  Switch->setMetadata("revamb.block.type",
                      QMD.tuple((uint32_t) DispatcherBlock));

  Dispatcher = Entry;
  DispatcherSwitch = Switch;
  NoReturn.setDispatcher(Dispatcher);

  // Create basic blocks to handle jumps to any PC and to a PC we didn't expect
  AnyPC = BasicBlock::Create(Context, "anypc", OutputFunction);
  UnexpectedPC = BasicBlock::Create(Context, "unexpectedpc", OutputFunction);

  setCFGForm(CFGForm::SemanticPreservingCFG);
}

static void purge(BasicBlock *BB) {
  // Allow up to a single instruction in the basic block
  if (!BB->empty())
    BB->begin()->eraseFromParent();
  revng_assert(BB->empty());
}

std::set<BasicBlock *> JumpTargetManager::computeUnreachable() {
  ReversePostOrderTraversal<BasicBlock *> RPOT(&TheFunction->getEntryBlock());
  std::set<BasicBlock *> Reachable;
  for (BasicBlock *BB : RPOT)
    Reachable.insert(BB);

  // TODO: why is isTranslatedBB(&BB) necessary?
  std::set<BasicBlock *> Unreachable;
  for (BasicBlock &BB : *TheFunction)
    if (Reachable.count(&BB) == 0 and isTranslatedBB(&BB))
      Unreachable.insert(&BB);

  return Unreachable;
}

void JumpTargetManager::setCFGForm(CFGForm::Values NewForm) {
  revng_assert(CurrentCFGForm != NewForm);
  revng_assert(NewForm != CFGForm::UnknownFormCFG);

  std::set<BasicBlock *> Unreachable;

  CFGForm::Values OldForm = CurrentCFGForm;
  CurrentCFGForm = NewForm;

  switch (NewForm) {
  case CFGForm::SemanticPreservingCFG:
    purge(AnyPC);
    BranchInst::Create(dispatcher(), AnyPC);
    // TODO: Here we should have an hard fail, since it's the situation in
    //       which we expected to know where execution could go but we made a
    //       mistake.
    purge(UnexpectedPC);
    BranchInst::Create(dispatcher(), UnexpectedPC);
    break;

  case CFGForm::RecoveredOnlyCFG:
  case CFGForm::NoFunctionCallsCFG:
    purge(AnyPC);
    new UnreachableInst(Context, AnyPC);
    purge(UnexpectedPC);
    new UnreachableInst(Context, UnexpectedPC);
    break;

  default:
    revng_abort("Not implemented yet");
  }

  QuickMetadata QMD(Context);
  AnyPC->getTerminator()->setMetadata("revamb.block.type",
                                      QMD.tuple((uint32_t) AnyPCBlock));
  TerminatorInst *UnexpectedPCJump = UnexpectedPC->getTerminator();
  UnexpectedPCJump->setMetadata("revamb.block.type",
                                QMD.tuple((uint32_t) UnexpectedPCBlock));

  // If we're entering or leaving the NoFunctionCallsCFG form, update all the
  // branch instruction forming a function call
  if (NewForm == CFGForm::NoFunctionCallsCFG
      || OldForm == CFGForm::NoFunctionCallsCFG) {
    if (auto *FunctionCall = TheModule.getFunction("function_call")) {
      for (User *U : FunctionCall->users()) {
        auto *Call = cast<CallInst>(U);

        // Ignore indirect calls
        // TODO: why this is needed is unclear
        if (isa<ConstantPointerNull>(Call->getArgOperand(0)))
          continue;

        auto *Terminator = cast<TerminatorInst>(nextNonMarker(Call));
        revng_assert(Terminator->getNumSuccessors() == 1);

        // Get the correct argument, the first is the callee, the second the
        // return basic block
        int OperandIndex = NewForm == CFGForm::NoFunctionCallsCFG ? 1 : 0;
        Value *Op = Call->getArgOperand(OperandIndex);
        BasicBlock *NewSuccessor = cast<BlockAddress>(Op)->getBasicBlock();
        Terminator->setSuccessor(0, NewSuccessor);
      }
    }
  }

  rebuildDispatcher();

  if (Verify.isEnabled()) {
    Unreachable = computeUnreachable();
    if (Unreachable.size() != 0) {
      Verify << "The following basic blocks are unreachable after setCFGForm("
             << CFGForm::getName(NewForm) << "):\n";
      for (BasicBlock *BB : Unreachable) {
        Verify << "  " << getName(BB) << " (predecessors:";
        for (BasicBlock *Predecessor : make_range(pred_begin(BB), pred_end(BB)))
          Verify << " " << getName(Predecessor);

        if (uint64_t PC = getBasicBlockPC(BB)) {
          auto It = JumpTargets.find(PC);
          if (It != JumpTargets.end()) {
            Verify << ", reasons:";
            for (const char *Reason : It->second.getReasonNames())
              Verify << " " << Reason;
          }
        }

        Verify << ")\n";
      }
      revng_abort();
    }
  }
}

void JumpTargetManager::rebuildDispatcher() {
  // Remove all cases
  unsigned NumCases = DispatcherSwitch->getNumCases();
  while (NumCases-- > 0)
    DispatcherSwitch->removeCase(DispatcherSwitch->case_begin());

  auto *PCRegType = PCReg->getType()->getPointerElementType();
  auto *SwitchType = cast<IntegerType>(PCRegType);

  // Add all the jump targets if we're using the SemanticPreservingCFG, or
  // only those with no predecessors otherwise
  for (auto &P : JumpTargets) {
    uint64_t PC = P.first;
    BasicBlock *BB = P.second.head();
    if (CurrentCFGForm == CFGForm::SemanticPreservingCFG
        || !hasPredecessors(BB))
      DispatcherSwitch->addCase(ConstantInt::get(SwitchType, PC), BB);
  }

  //
  // Make sure every generated basic block is reachable
  //
  if (CurrentCFGForm != CFGForm::SemanticPreservingCFG) {
    // Compute the set of reachable jump targets
    OnceQueue<BasicBlock *> WorkList;
    for (BasicBlock *BB : DispatcherSwitch->successors())
      WorkList.insert(BB);

    while (not WorkList.empty()) {
      BasicBlock *BB = WorkList.pop();
      for (BasicBlock *Successor : make_range(succ_begin(BB), succ_end(BB)))
        WorkList.insert(Successor);
    }

    std::set<BasicBlock *> Reachable = WorkList.visited();

    // Identify all the unreachable jump targets
    for (auto &P : JumpTargets) {
      uint64_t PC = P.first;
      const JumpTarget &JT = P.second;
      BasicBlock *BB = JT.head();

      // Add to the switch all the unreachable jump targets whose reason is not
      // just direct jump
      if (Reachable.count(BB) == 0
          and not JT.isOnlyReason(JTReason::DirectJump)) {
        DispatcherSwitch->addCase(ConstantInt::get(SwitchType, PC), BB);
      }
    }
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
    for (uint64_t PC : SimpleLiterals)
      registerJT(PC, JTReason::SimpleLiteral);
    SimpleLiterals.clear();
  }

  if (empty()) {
    // Purge all the generated basic blocks without predecessors
    std::vector<BasicBlock *> ToDelete;
    for (BasicBlock &BB : *TheFunction) {
      if (isTranslatedBB(&BB) and &BB != &TheFunction->getEntryBlock()
          and pred_begin(&BB) == pred_end(&BB)) {
        revng_assert(getBasicBlockPC(&BB) == 0);
        ToDelete.push_back(&BB);
      }
    }
    for (BasicBlock *BB : ToDelete)
      BB->eraseFromParent();

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

    if (Verify.isEnabled())
      revng_assert(not verifyModule(TheModule, &dbgs()));

    revng_log(JTCountLog, "Harvesting: SROA, ConstProp, EarlyCSE and SET");

    legacy::FunctionPassManager OptimizingPM(&TheModule);
    OptimizingPM.add(createSROAPass());
    OptimizingPM.add(createConstantPropagationPass());
    OptimizingPM.add(createEarlyCSEPass());
    OptimizingPM.run(*TheFunction);

    legacy::FunctionPassManager PreliminaryBranchesPM(&TheModule);
    PreliminaryBranchesPM.add(new TranslateDirectBranchesPass(this));
    PreliminaryBranchesPM.run(*TheFunction);

    // TODO: eventually, `setCFGForm` should be replaced by using a CustomCFG
    // To improve the quality of our analysis, keep in the CFG only the edges we
    // where able to recover (e.g., no jumps to the dispatcher)
    setCFGForm(CFGForm::RecoveredOnlyCFG);

    NewBranches = 0;
    legacy::FunctionPassManager AnalysisPM(&TheModule);
    AnalysisPM.add(new SETPass(this, false, &Visited));
    AnalysisPM.add(new TranslateDirectBranchesPass(this));
    AnalysisPM.run(*TheFunction);

    // Restore the CFG
    setCFGForm(CFGForm::SemanticPreservingCFG);

    revng_log(JTCountLog,
              std::dec << Unexplored.size() << " new jump targets and "
                       << NewBranches << " new branches were found");
  }

  if (not NoOSRA && empty()) {
    if (Verify.isEnabled())
      revng_assert(not verifyModule(TheModule, &dbgs()));

    NoReturn.registerSyscalls(TheFunction);

    do {

      revng_log(JTCountLog,
                "Harvesting: reset Visited, "
                  << (NewBranches > 0 ? "SROA, ConstProp, EarlyCSE, " : "")
                  << "SET + OSRA");

      // TODO: decide what to do with Visited
      Visited.clear();
      if (NewBranches > 0) {
        legacy::FunctionPassManager OptimizingPM(&TheModule);
        OptimizingPM.add(createSROAPass());
        OptimizingPM.add(createConstantPropagationPass());
        OptimizingPM.add(createEarlyCSEPass());
        OptimizingPM.run(*TheFunction);
      }

      legacy::FunctionPassManager FunctionCallPM(&TheModule);
      FunctionCallPM.add(new FunctionCallIdentification());
      FunctionCallPM.run(*TheFunction);

      createJTReasonMD();

      setCFGForm(CFGForm::RecoveredOnlyCFG);

      NewBranches = 0;
      legacy::FunctionPassManager AnalysisPM(&TheModule);
      AnalysisPM.add(new SETPass(this, true, &Visited));
      AnalysisPM.add(new TranslateDirectBranchesPass(this));
      AnalysisPM.run(*TheFunction);

      // Restore the CFG
      setCFGForm(CFGForm::SemanticPreservingCFG);

      revng_log(JTCountLog,
                std::dec << Unexplored.size() << " new jump targets and "
                         << NewBranches << " new branches were found");

    } while (empty() && NewBranches > 0);
  }

  if (empty()) {
    revng_log(JTCountLog, "We're done looking for jump targets");
  }
}

using BlockWithAddress = JumpTargetManager::BlockWithAddress;
using JTM = JumpTargetManager;
const BlockWithAddress JTM::NoMoreTargets = BlockWithAddress(0, nullptr);
