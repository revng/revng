/// \file JumpTargetManager.cpp
/// \brief This file handles the possible jump targets encountered during
///        translation and the creation and management of the respective
///        BasicBlock.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdint>
#include <fstream>
#include <queue>
#include <sstream>

// Boost includes
#include <boost/icl/interval_set.hpp>
#include <boost/icl/right_open_interval.hpp>
#include <boost/type_traits/is_same.hpp>

// LLVM includes
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/CodeGen/UnreachableBlockElim.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/JumpThreading.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

// Local libraries includes
#include "revng/ADT/Queue.h"
#include "revng/BasicAnalyses/AdvancedValueInfo.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/BasicAnalyses/ShrinkInstructionOperandsPass.h"
#include "revng/Support/Assert.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/revng.h"

// Local includes
#include "AdvancedValueInfoPass.h"
#include "CPUStateAccessAnalysisPass.h"
#include "DropHelperCallsPass.h"
#include "JumpTargetManager.h"
#include "SubGraph.h"

using namespace llvm;

namespace {

Logger<> JTCountLog("jtcount");
Logger<> NewEdgesLog("new-edges");
Logger<> Verify("verify");
Logger<> RegisterJTLog("registerjt");

RegisterPass<TranslateDirectBranchesPass> X("translate-db",
                                            "Translate Direct Branches"
                                            " Pass",
                                            false,
                                            false);

} // namespace

char TranslateDirectBranchesPass::ID = 0;

void TranslateDirectBranchesPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<DominatorTreeWrapperPass>();
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

using TDBP = TranslateDirectBranchesPass;
void TDBP::pinPCStore(StoreInst *PCWrite,
                      bool Approximate,
                      const std::vector<uint64_t> &Destinations) {
  Function &F = *PCWrite->getParent()->getParent();
  LLVMContext &Context = getContext(&F);
  Value *PCReg = JTM->pcReg();
  auto *RegType = cast<IntegerType>(PCReg->getType()->getPointerElementType());
  auto C = [RegType](uint64_t A) { return ConstantInt::get(RegType, A); };
  BasicBlock *AnyPC = JTM->anyPC();
  BasicBlock *UnexpectedPC = JTM->unexpectedPC();

  // We don't care if we already handled this call too exitTB in the past,
  // information should become progressively more precise, so let's just remove
  // everything after this call and put a new handler
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
    Builder.CreateCondBr(Comparison, JTM->getBlockAt(Destinations[0]), FailBB);
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

bool TranslateDirectBranchesPass::pinAVIResults(Function &F) {
  QuickMetadata QMD(getContext(&F));

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *T = dyn_cast_or_null<MDTuple>(I.getMetadata("revng.avi"))) {
        StringRef TITMD = QMD.extract<StringRef>(T, 0);
        auto TIT = TrackedInstructionType::fromName(TITMD);
        auto *ValuesTuple = QMD.extract<MDTuple *>(T, 1);
        if (TIT == TrackedInstructionType::PCStore
            and ValuesTuple->getNumOperands() > 0) {
          std::vector<uint64_t> Values;
          Values.reserve(ValuesTuple->getNumOperands());
          for (const MDOperand &Operand : ValuesTuple->operands())
            Values.push_back(QMD.extract<uint64_t>(Operand.get()));
          pinPCStore(cast<StoreInst>(&I), false, Values);
        }
      }
    }
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
        if (auto *Callee = cast<Function>(skipCasts(Call->getCalledValue()))) {
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

bool TranslateDirectBranchesPass::runOnModule(Module &M) {
  Function &F = *M.getFunction("root");
  pinConstantStore(F);
  pinAVIResults(F);
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

MaterializedValue
JumpTargetManager::readFromPointer(Constant *Pointer, BinaryFile::Endianess E) {
  Type *LoadedType = Pointer->getType()->getPointerElementType();
  const DataLayout &DL = TheModule.getDataLayout();
  unsigned LoadSize = DL.getTypeSizeInBits(LoadedType) / 8;
  uint64_t LoadAddress = getZExtValue(cast<ConstantInt>(skipCasts(Pointer)),
                                      DL);
  UnusedCodePointers.erase(LoadAddress);
  registerReadRange(LoadAddress, LoadSize);

  // Prevent overflow when computing the label interval
  if (LoadAddress + LoadSize < LoadAddress) {
    return MaterializedValue::invalid();
  }

  const auto &Labels = binary().labels();
  using interval = boost::icl::interval<uint64_t>;
  auto Interval = interval::right_open(LoadAddress, LoadAddress + LoadSize);
  auto It = Labels.find(Interval);
  if (It != Labels.end()) {
    const Label *Match = nullptr;
    for (const Label *Candidate : It->second) {
      if (Candidate->size() == LoadSize
          and (Candidate->isAbsoluteValue() or Candidate->isBaseRelativeValue()
               or Candidate->isSymbolRelativeValue())) {
        revng_assert(Match == nullptr,
                     "Multiple value labels at the same location");
        Match = Candidate;
      }
    }

    if (Match != nullptr) {
      switch (Match->type()) {
      case LabelType::AbsoluteValue:
        return { Match->value() };

      case LabelType::BaseRelativeValue:
        return { binary().relocate(Match->value()) };

      case LabelType::SymbolRelativeValue:
        return { Match->symbolName(), Match->offset() };

      default:
        revng_abort();
      }
    }
  }

  // No labels found, fall back to read the raw value, if available
  Optional<uint64_t> Value = Binary.readRawValue(LoadAddress, LoadSize, E);

  if (Value)
    return { *Value };
  else
    return {};
}

template<typename T>
static cl::opt<T> *
getOption(StringMap<cl::Option *> &Options, const char *Name) {
  return static_cast<cl::opt<T> *>(Options[Name]);
}

JumpTargetManager::JumpTargetManager(Function *TheFunction,
                                     Value *PCReg,
                                     const BinaryFile &Binary,
                                     CSAAFactory createCSAA) :
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
  CurrentCFGForm(CFGForm::UnknownFormCFG),
  createCSAA(createCSAA) {

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
          auto *Callee = cast<Function>(skipCasts(Call->getCalledValue()));
          revng_assert(Callee->getName() != "newpc");
          if (Callee == ExitTB) {
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
  class Visitor : public BackwardBFSVisitor<Visitor> {
  private:
    Value *PCReg;
    Instruction *Skip;
    StoreInst *Result;

  public:
    Visitor(Value *PCReg, Instruction *Skip) :
      PCReg(PCReg),
      Skip(Skip),
      Result(nullptr) {}

    VisitAction visit(instruction_range Range) {
      for (Instruction &I : Range) {
        // Stop at helpers/newpc
        if (isa<CallInst>(&I) and &I != Skip)
          return NoSuccessors;

        auto *Store = dyn_cast<StoreInst>(&I);
        if (Store != nullptr && Store->getPointerOperand() == PCReg) {

          // If Result is not null, it's the second store to pc we find
          if (Result != nullptr) {
            Result = nullptr;
            return StopNow;
          }

          Result = Store;
          return NoSuccessors;
        }
      }

      return Continue;
    }

    StoreInst *getResult() const { return Result; }
  };

  Visitor V(PCReg, TheInstruction);
  V.run(TheInstruction);
  return V.getResult();
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
  setBlockType(FailUnreachable, BlockType::DispatcherFailureBlock);

  // Switch on the first argument of the function
  Builder.SetInsertPoint(Entry);
  Value *SwitchOn = Builder.CreateLoad(SwitchOnPtr);
  SwitchInst *Switch = Builder.CreateSwitch(SwitchOn, DispatcherFail);

  // The switch is the terminator of the dispatcher basic block
  setBlockType(Switch, BlockType::DispatcherBlock);

  Dispatcher = Entry;
  DispatcherSwitch = Switch;

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

  setBlockType(AnyPC->getTerminator(), BlockType::AnyPCBlock);

  TerminatorInst *UnexpectedPCJump = UnexpectedPC->getTerminator();
  setBlockType(UnexpectedPCJump, BlockType::UnexpectedPCBlock);

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

/// \brief Simple pass to drop `range` metadata, which is sometimes detrimental
class DropRangeMetadataPass : public PassInfoMixin<DropRangeMetadataPass> {

public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        I.setMetadata("range", nullptr);
    return PreservedAnalyses::none();
  }
};

/// \brief Drop all the call to marker functions
class DropMarkerCalls : public PassInfoMixin<DropMarkerCalls> {
private:
  SmallVector<StringRef, 4> ToPreserve;

public:
  DropMarkerCalls(SmallVector<StringRef, 4> ToPreserve) :
    ToPreserve(ToPreserve) {}

public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    Module *M = F.getParent();
    std::vector<CallInst *> ToErase;

    for (StringRef MarkerName : MarkerFunctionNames) {
      if (Function *Marker = M->getFunction(MarkerName)) {

        //
        // Check if we should preserve this marker
        //
        auto It = std::find(ToPreserve.begin(), ToPreserve.end(), MarkerName);
        auto End = std::end(ToPreserve);
        if (It != End) {
          Marker->setDoesNotReturn();
          continue;
        }

        //
        // Register all the calls to be erased
        //
        for (User *U : Marker->users())
          if (auto *Call = dyn_cast<CallInst>(U))
            if (Call->getParent()->getParent() == &F)
              ToErase.push_back(Call);
      }
    }

    //
    // Actually drop the calls
    //
    for (CallInst *Call : ToErase)
      Call->eraseFromParent();

    return PreservedAnalyses::none();
  }
};

void JumpTargetManager::aliasAnalysis() {
  unsigned AliasScopeMDKindID = TheModule.getMDKindID("alias.scope");
  unsigned NoAliasMDKindID = TheModule.getMDKindID("noalias");

  LLVMContext &Context = TheModule.getContext();
  QuickMetadata QMD(Context);
  MDBuilder MDB(Context);
  MDNode *CSVDomain = MDB.createAliasScopeDomain("CSVAliasDomain");

  struct CSVAliasInfo {
    MDNode *AliasScope;
    MDNode *AliasSet;
    MDNode *NoAliasSet;
  };
  std::map<const GlobalVariable *, CSVAliasInfo> CSVAliasInfoMap;

  std::vector<GlobalVariable *> CSVs;
  NamedMDNode *NamedMD = TheModule.getOrInsertNamedMetadata("revng.csv");
  auto *Tuple = cast<MDTuple>(NamedMD->getOperand(0));
  for (const MDOperand &Operand : Tuple->operands()) {
    auto *CSV = cast<GlobalVariable>(QMD.extract<Constant *>(Operand.get()));
    CSVs.push_back(CSV);
  }

  // Build alias scopes
  std::vector<Metadata *> AllCSVScopes;
  for (const GlobalVariable *CSV : CSVs) {
    CSVAliasInfo &AliasInfo = CSVAliasInfoMap[CSV];

    std::string Name = CSV->getName();
    MDNode *CSVScope = MDB.createAliasScope(Name, CSVDomain);
    AliasInfo.AliasScope = CSVScope;
    AllCSVScopes.push_back(CSVScope);
    MDNode *CSVAliasSet = MDNode::get(Context,
                                      ArrayRef<Metadata *>({ CSVScope }));
    AliasInfo.AliasSet = CSVAliasSet;
  }
  MDNode *MemoryAliasSet = MDNode::get(Context, AllCSVScopes);

  // Build noalias sets
  for (const GlobalVariable *CSV : CSVs) {
    CSVAliasInfo &AliasInfo = CSVAliasInfoMap[CSV];
    std::vector<Metadata *> OtherCSVScopes;
    for (const auto &Q : CSVAliasInfoMap)
      if (Q.first != CSV)
        OtherCSVScopes.push_back(Q.second.AliasScope);

    MDNode *CSVNoAliasSet = MDNode::get(Context, OtherCSVScopes);
    AliasInfo.NoAliasSet = CSVNoAliasSet;
  }

  // Decorate the IR with alias information
  for (Function &F : TheModule) {
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        Value *Ptr = nullptr;

        if (auto *L = dyn_cast<LoadInst>(&I))
          Ptr = L->getPointerOperand();
        else if (auto *S = dyn_cast<StoreInst>(&I))
          Ptr = S->getPointerOperand();
        else
          continue;

        // Check if the pointer is a CSV
        if (auto *GV = dyn_cast<GlobalVariable>(Ptr)) {
          auto It = CSVAliasInfoMap.find(GV);
          if (It != CSVAliasInfoMap.end()) {
            // Set alias.scope and noalias metadata
            I.setMetadata(AliasScopeMDKindID, It->second.AliasSet);
            I.setMetadata(NoAliasMDKindID, It->second.NoAliasSet);
            continue;
          }
        }

        // It's not a CSV memory access, set noalias info
        I.setMetadata(NoAliasMDKindID, MemoryAliasSet);
      }
    }
  }
}

void JumpTargetManager::harvestWithAVI() {
  Module *M = TheFunction->getParent();

  //
  // Update alias analysis
  //
  aliasAnalysis();

  //
  // Update CPUStateAccessAnalysisPass
  //
  legacy::PassManager PM;
  PM.add(createCSAA());
  PM.run(TheModule);

  //
  // Collect all the CSVs
  //
  std::set<GlobalVariable *> CSVs;
  QuickMetadata QMD(Context);
  NamedMDNode *NamedMD = TheModule.getOrInsertNamedMetadata("revng.csv");
  auto *Tuple = cast<MDTuple>(NamedMD->getOperand(0));
  for (const MDOperand &Operand : Tuple->operands()) {
    auto *CSV = cast<GlobalVariable>(QMD.extract<Constant *>(Operand.get()));
    CSVs.insert(CSV);
  }

  //
  // Clone the root function
  //

  // Prune the dispatcher
  setCFGForm(CFGForm::RecoveredOnlyCFG);

  revng_assert(computeUnreachable().size() == 0);

  // Clone the function
  Function *OptimizedFunction = nullptr;
  ValueToValueMapTy OldToNew;
  OptimizedFunction = CloneFunction(TheFunction, OldToNew);

  //
  // Identify all the calls to exitTB
  //
  std::map<uint32_t, Instruction *> AVIIDToOld;
  uint32_t AVIID = 0;

  Function *ExitTB = M->getFunction("exitTB");
  for (User *U : ExitTB->users()) {
    if (auto *Call = dyn_cast<CallInst>(U)) {
      BasicBlock *BB = Call->getParent();
      if (BB->getParent() == TheFunction) {
        // Find the last PC write
        StoreInst *Store = getPrevPCWrite(Call);
        if (Store != nullptr) {
          auto *NewStore = cast<StoreInst>(OldToNew[Store]);
          if (NewStore->getMetadata("revng.avi") == nullptr) {
            NewStore->setMetadata("revng.avi.mark", QMD.tuple(AVIID));
            AVIIDToOld[AVIID] = Store;
            ++AVIID;
          }
        }
      }
    }
  }

  // Register instructions reading memory
  for (BasicBlock &BB : *TheFunction) {
    for (Instruction &I : BB) {

      Value *Pointer = nullptr;
      if (auto *Load = dyn_cast<LoadInst>(&I))
        Pointer = Load->getPointerOperand();
      else if (auto *Store = dyn_cast<StoreInst>(&I))
        Pointer = Store->getPointerOperand();

      if (Pointer != nullptr and isMemory(Pointer)) {
        auto *NewI = cast<Instruction>(OldToNew[&I]);
        if (NewI->getMetadata("revng.avi") == nullptr) {
          NewI->setMetadata("revng.avi.mark", QMD.tuple(AVIID));
          AVIIDToOld[AVIID] = &I;
          ++AVIID;
        }
      }
    }
  }

  // Restore the dispatcher
  setCFGForm(CFGForm::SemanticPreservingCFG);

  revng_assert(computeUnreachable().size() == 0);

  //
  // Create an alloca per CSV (except for the PC)
  //
  IRBuilder<> AllocaBuilder(&*OptimizedFunction->getEntryBlock().begin());
  auto *Marker = AllocaBuilder.CreateLoad(PCReg);
  Instruction *T = OptimizedFunction->getEntryBlock().getTerminator();
  IRBuilder<> InitializeBuilder(T);
  std::map<GlobalVariable *, AllocaInst *> CSVMap;

  for (GlobalVariable *CSV : CSVs) {

    if (CSV == PCReg)
      continue;

    Type *CSVType = CSV->getType()->getPointerElementType();
    auto *Alloca = AllocaBuilder.CreateAlloca(CSVType, nullptr, CSV->getName());
    CSVMap[CSV] = Alloca;

    // Replace all uses of the CSV within OptimizedFunction with the alloca
    auto UI = CSV->use_begin();
    auto E = CSV->use_end();
    while (UI != E) {
      Use &U = *UI;
      ++UI;

      if (auto *I = dyn_cast<Instruction>(U.getUser()))
        if (I->getParent()->getParent() == OptimizedFunction)
          U.set(Alloca);
    }

    InitializeBuilder.CreateStore(InitializeBuilder.CreateLoad(CSV), Alloca);
  }

  Marker->eraseFromParent();

  //
  // Helper to intrinsic promotion
  //
  IRBuilder<> Builder(Context);
  using MapperFunction = std::function<Instruction *(CallInst *)>;
  std::pair<const char *, MapperFunction> Mapping[] = {
    { "helper_clz",
      [&Builder](CallInst *Call) {
        return Builder.CreateBinaryIntrinsic(Intrinsic::ctlz,
                                             Call->getArgOperand(0),
                                             Builder.getFalse());
      } }
  };

  for (auto &P : Mapping) {
    const char *HelperName = P.first;
    auto Mapper = P.second;
    if (Function *Original = M->getFunction(HelperName)) {

      SmallVector<std::pair<Instruction *, Instruction *>, 16> Replacements;
      for (User *U : Original->users()) {
        if (auto *Call = dyn_cast<CallInst>(U)) {
          if (Call->getParent()->getParent() == OptimizedFunction) {
            Builder.SetInsertPoint(Call);
            Instruction *NewI = Mapper(Call);
            NewI->copyMetadata(*Call);
            Replacements.emplace_back(Call, NewI);
          }
        }
      }

      // Apply replacements
      for (auto &P : Replacements) {
        P.first->replaceAllUsesWith(P.second);
        P.first->eraseFromParent();
      }
    }
  }

  //
  // Optimize the hell out of it and collect the possible values of indirect
  // branches
  //

  StringRef SyscallHelperName = Binary.architecture().syscallHelper();
  Function *SyscallHelper = M->getFunction(SyscallHelperName);
  StringRef SyscallIDCSVName = Binary.architecture().syscallNumberRegister();
  GlobalVariable *SyscallIDCSV = M->getGlobalVariable(SyscallIDCSVName);

  SummaryCallsBuilder SCB(CSVMap);

  FunctionPassManager FPM;
  FPM.addPass(DropMarkerCalls({ "exitTB" }));
  FPM.addPass(DropHelperCallsPass(SyscallHelper, SyscallIDCSV, SCB));
  FPM.addPass(ShrinkInstructionOperandsPass());
  FPM.addPass(PromotePass());
  FPM.addPass(InstCombinePass());
  FPM.addPass(JumpThreadingPass());
  FPM.addPass(UnreachableBlockElimPass());
  FPM.addPass(InstCombinePass());
  FPM.addPass(EarlyCSEPass(true));
  FPM.addPass(DropRangeMetadataPass());
  FPM.addPass(AdvancedValueInfoPass(this));

  FunctionAnalysisManager FAM;
  FAM.registerPass([] {
    AAManager AA;
    AA.registerFunctionAnalysis<ScopedNoAliasAA>();
    return AA;
  });

  PassBuilder PB;
  PB.registerFunctionAnalyses(FAM);

  FPM.run(*OptimizedFunction, FAM);

  revng_assert(not verifyModule(*OptimizedFunction->getParent(), &dbgs()));

  //
  // Collect the results
  //
  for (BasicBlock &BB : *OptimizedFunction) {
    for (Instruction &I : BB) {
      auto *T = dyn_cast_or_null<MDTuple>(I.getMetadata("revng.avi"));
      auto *Marker = dyn_cast_or_null<MDTuple>(I.getMetadata("revng.avi.mark"));
      if (T != nullptr and Marker != nullptr) {
        revng_assert(T->getNumOperands() == 2);
        revng_assert(Marker->getNumOperands() == 1);

        uint32_t AVIID = QMD.extract<uint32_t>(Marker, 0);

        StringRef TITMD = QMD.extract<StringRef>(T, 0);
        auto TIT = TrackedInstructionType::fromName(TITMD);
        Instruction *InstructionInRoot = AVIIDToOld[AVIID];
        auto *NewValuesMD = QMD.extract<MDTuple *>(T, 1);

        bool AllGood = true;
        for (const MDOperand &Operand : NewValuesMD->operands()) {
          uint64_t Value = QMD.extract<uint64_t>(Operand.get());
          if (not isPC(Value)) {
            AllGood = false;
            break;
          }
        }

        if (not AllGood)
          continue;

        auto OperandsCount = NewValuesMD->getNumOperands();
        if (OperandsCount != 0 and NewEdgesLog.isEnabled()) {
          NewEdgesLog << OperandsCount << " targets from " << getName(&I)
                      << "\n"
                      << DoLog;
        }

        auto *AVIMD = InstructionInRoot->getMetadata("revng.avi");
        if (auto *Old = dyn_cast_or_null<MDTuple>(AVIMD)) {
          // Merge the already present values with the new ones
          std::set<uint64_t> Values;
          for (const MDOperand &Operand :
               QMD.extract<MDTuple *>(Old, 1)->operands())
            Values.insert(QMD.extract<uint64_t>(Operand.get()));
          for (const MDOperand &Operand : NewValuesMD->operands())
            Values.insert(QMD.extract<uint64_t>(Operand.get()));
          std::vector<Metadata *> ValuesMD;
          ValuesMD.reserve(Values.size());
          for (uint64_t V : Values)
            ValuesMD.push_back(QMD.get(V));
          T = QMD.tuple({ QMD.get(TrackedInstructionType::getName(TIT)),
                          QMD.tuple(ValuesMD) });
        } else {
          T = QMD.tuple({ QMD.get(TrackedInstructionType::getName(TIT)),
                          QMD.extract<MDTuple *>(T, 1) });
        }

        InstructionInRoot->setMetadata("revng.avi", T);

        auto *Values = QMD.extract<llvm::MDTuple *>(T, 1);
        if (TIT == TrackedInstructionType::PCStore
            or TIT == TrackedInstructionType::MemoryStore) {
          bool IsMemoryStore = TIT == TrackedInstructionType::MemoryStore;
          JTReason::Values Reason = (IsMemoryStore ? JTReason::MemoryStore :
                                                     JTReason::PCStore);
          for (const MDOperand &Operand : Values->operands()) {
            uint64_t Address = QMD.extract<uint64_t>(Operand.get());
            registerJT(Address, Reason);
          }
        } else if (TIT == TrackedInstructionType::MemoryLoad) {
          for (const MDOperand &Operand : Values->operands()) {
            uint64_t Address = QMD.extract<uint64_t>(Operand.get());
            markJT(Address, JTReason::LoadAddress);
          }
        }
      }
    }
  }

  //
  // Drop the optimized function
  //
  OptimizedFunction->eraseFromParent();

  // Drop temporary functions
  SCB.cleanup();
}

// Harvesting proceeds trying to avoid to run expensive analyses if not strictly
// necessary. To do this we keep in mind two aspects: do we have new basic
// blocks to visit? If so, we avoid any further anyalysis and give back control
// to the translator. If not, we proceed with other analyses until we either
// find a new basic block to translate. If we can't find a new block to
// translate we proceed as long as we are able to create new edges on the CFG
// (not considering the dispatcher).
void JumpTargetManager::harvest() {

  if (empty()) {
    revng_log(JTCountLog, "Collecting simple literals");
    for (uint64_t PC : SimpleLiterals)
      registerJT(PC, JTReason::SimpleLiteral);
    SimpleLiterals.clear();
  }

  if (empty()) {
    revng_log(JTCountLog, "Collecting simple literals");

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

    revng_log(JTCountLog, "Preliminary harvesting");

    legacy::FunctionPassManager OptimizingPM(&TheModule);
    OptimizingPM.add(createSROAPass());
    OptimizingPM.run(*TheFunction);

    legacy::PassManager PreliminaryBranchesPM;
    PreliminaryBranchesPM.add(new TranslateDirectBranchesPass(this));
    PreliminaryBranchesPM.run(TheModule);

    if (empty()) {
      revng_log(JTCountLog, "Harvesting with Advanced Value Info");
      harvestWithAVI();
    }

    // TODO: eventually, `setCFGForm` should be replaced by using a CustomCFG
    // To improve the quality of our analysis, keep in the CFG only the edges we
    // where able to recover (e.g., no jumps to the dispatcher)
    setCFGForm(CFGForm::RecoveredOnlyCFG);

    NewBranches = 0;
    legacy::PassManager AnalysisPM;
    AnalysisPM.add(new TranslateDirectBranchesPass(this));
    AnalysisPM.run(TheModule);

    // Restore the CFG
    setCFGForm(CFGForm::SemanticPreservingCFG);

    if (JTCountLog.isEnabled()) {
      JTCountLog << std::dec << Unexplored.size() << " new jump targets and "
                 << NewBranches << " new branches were found" << DoLog;
    }
  }

  if (empty()) {
    revng_log(JTCountLog, "We're done looking for jump targets");
  }
}

using BlockWithAddress = JumpTargetManager::BlockWithAddress;
using JTM = JumpTargetManager;
const BlockWithAddress JTM::NoMoreTargets = BlockWithAddress(0, nullptr);
