/// \file JumpTargetManager.cpp
/// This file handles the possible jump targets encountered during translation
/// and the creation and management of the respective BasicBlock.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/Progress.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "revng/Lift/Lift.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/Statistics.h"

#include "JumpTargetManager.h"
#include "RootAnalyzer.h"
#include "SubGraph.h"

RegisterIRHelper JumpToSymbolMarker("jump_to_symbol", "absent after lift");
RegisterIRHelper ExitTBMarker("exitTB", "absent after lift");

using namespace llvm;

namespace {

Logger<> JTCountLog("jtcount");
Logger<> RegisterJTLog("registerjt");

CounterMap<std::string> HarvestingStats("harvesting");

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

static void exitTBCleanup(Instruction *ExitTBCall) {
  // TODO: for some reason we don't always have a terminator
  if (auto *T = nextNonMarker(ExitTBCall))
    eraseFromParent(T);
}

using TDBP = TranslateDirectBranchesPass;

TDBP::TranslateDirectBranchesPass(JumpTargetManager *J) :
  ModulePass(ID), JTM(J), PCH(J->programCounterHandler()) {
}

using DispatcherTargets = ProgramCounterHandler::DispatcherTargets;
void TDBP::pinExitTB(CallInst *ExitTBCall, DispatcherTargets &Destinations) {
  revng_assert(ExitTBCall != nullptr);
  revng_assert(Destinations.size() != 0);

  LLVMContext &Context = getContext(ExitTBCall);
  BasicBlock *AnyPC = JTM->anyPC();
  BasicBlock *UnexpectedPC = JTM->unexpectedPC();
  BasicBlock *Source = ExitTBCall->getParent();

  using CI = ConstantInt;
  auto *ExitTBArg = CI::get(Type::getInt32Ty(Context), Destinations.size());
  uint64_t OldTargetsCount = getLimitedValue(ExitTBCall->getArgOperand(0));

  // TODO: we should check Destinations.size() >= OldTargetsCount
  // TODO: we should also check the destinations are actually the same

  BasicBlock *BB = ExitTBCall->getParent();

  if (auto *Dispatcher = dyn_cast<SwitchInst>(BB->getTerminator()))
    PCH->destroyDispatcher(Dispatcher);

  // Kill everything is after the call to exitTB
  exitTBCleanup(ExitTBCall);

  // Mark this call to exitTB as handled
  ExitTBCall->setArgOperand(0, ExitTBArg);

  PCH->buildDispatcher(Destinations,
                       BB,
                       UnexpectedPC,
                       BlockType::IndirectBranchDispatcherHelperBlock);

  // Move all the markers right before the branch instruction
  Instruction *Last = BB->getTerminator();
  auto It = ExitTBCall->getIterator();
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
    JTM->recordNewBranches(Source, Destinations.size() - OldTargetsCount);
}

bool TDBP::pinMaterializedValues(Function &F) {
  QuickMetadata QMD(getContext(&F));
  Module *M = F.getParent();

  // Lazily create the `jump_to_symbol` marker
  auto *Marker = getIRHelper("jump_to_symbol", *M);
  if (Marker == nullptr) {
    LLVMContext &C = M->getContext();
    auto *FT = FunctionType::get(Type::getVoidTy(C),
                                 { Type::getInt8PtrTy(C) },
                                 false);
    Marker = createIRHelper("jump_to_symbol",
                            *M,
                            FT,
                            GlobalValue::ExternalLinkage);
    FunctionTags::Marker.addTo(Marker);
  }

  Function *ExitTB = JTM->exitTB();
  for (User *U : ExitTB->users()) {
    if (auto *Call = dyn_cast<CallInst>(U)) {
      if (Call->getParent()->getParent() != &F)
        continue;

      auto *MD = Call->getMetadata("revng.targets");
      if (auto *T = dyn_cast_or_null<MDTuple>(MD)) {
        // Compute the list of MetaAddress/symbols destinations
        SmallVector<llvm::StringRef> SymbolDestinations;
        SmallVector<MetaAddress> DirectDestinations;
        for (const MDOperand &Operand : T->operands()) {
          auto *Tuple = QMD.extract<MDTuple *>(Operand.get());
          auto SymbolName = QMD.extract<StringRef>(Tuple->getOperand(0).get());
          auto *Value = QMD.extract<ConstantInt *>(Tuple->getOperand(1).get());
          bool HasDynamicSymbol = SymbolName.size() != 0;
          if (HasDynamicSymbol) {
            SymbolDestinations.push_back(SymbolName);
          } else {
            auto Address = MetaAddress::decomposeIntegerPC(Value);
            revng_assert(Address.isValid());
            DirectDestinations.push_back(Address);
          }
        }

        // We handle two situations: all DirectDestinations or one symbol
        // destination
        bool HasDirectDestinations = DirectDestinations.size() > 0;
        bool HasSymbolDestinations = SymbolDestinations.size() > 0;
        bool HasOneSymbolDestination = SymbolDestinations.size() == 1;
        if (HasDirectDestinations and not HasSymbolDestinations) {
          // We have at least a direct destination, prepare a list of jump
          // targets
          ProgramCounterHandler::DispatcherTargets Values;
          Values.reserve(T->getNumOperands());
          for (const auto &Address : DirectDestinations)
            Values.emplace_back(Address, JTM->getBlockAt(Address));
          pinExitTB(Call, Values);
        } else if (HasOneSymbolDestination) {
          // Jump to a symbol

          auto *T = Call->getParent()->getTerminator();
          revng_assert(hasMarker(T, getCalledFunction(Call)));

          // Purge existing marker, if any
          if (CallInst *MarkerInstruction = getMarker(T, Marker))
            MarkerInstruction->eraseFromParent();

          // Create the marker
          StringRef SymbolName = SymbolDestinations[0];
          // TODO: in theory we could insert this before T, not Call, but it's
          //       violating some assumption somewhere
          CallInst::Create({ Marker },
                           { getUniqueString(M, SymbolName) },
                           {},
                           "",
                           Call);
        }
      }
    }
  }

  return true;
}

void TDBP::pinConstantStoreInternal(MetaAddress Address, CallInst *ExitTBCall) {
  revng_assert(Address.isValid());

  BasicBlock *TargetBlock = JTM->registerJT(Address, JTReason::DirectJump);

  const Module *M = getModule(ExitTBCall);
  LLVMContext &Context = getContext(M);

  // Remove unreachable right after the exit_tb
  BasicBlock::iterator CallIt(ExitTBCall);
  BasicBlock::iterator BlockEnd = ExitTBCall->getParent()->end();
  CallIt++;
  revng_assert(CallIt != BlockEnd and isa<UnreachableInst>(&*CallIt));
  eraseFromParent(&*CallIt);

  // Cleanup of what's afterwards (only a unconditional jump is
  // allowed)
  CallIt = BasicBlock::iterator(ExitTBCall);
  BlockEnd = ExitTBCall->getParent()->end();
  if (++CallIt != BlockEnd)
    purgeBranch(CallIt);

  if (TargetBlock != nullptr) {
    // A target was found, jump there
    BranchInst::Create(TargetBlock, ExitTBCall->getParent());
    JTM->recordNewBranches(ExitTBCall->getParent(), 1);
  } else {
    // We're jumping to an unknown or invalid location, jump back to the
    // dispatcher
    // TODO: emit a warning
    BranchInst::Create(JTM->unexpectedPC(), ExitTBCall);
  }

  eraseFromParent(ExitTBCall);
}

bool TDBP::pinConstantStore(Function &F) {
  auto ExitTB = JTM->exitTB();
  auto ExitTBIt = ExitTB->use_begin();
  while (ExitTBIt != ExitTB->use_end()) {
    // Take note of the use and increment the iterator immediately: this allows
    // us to erase the call to exit_tb without unexpected behaviors
    Use &ExitTBUse = *ExitTBIt++;
    auto *Call = cast<CallInst>(ExitTBUse.getUser());
    revng_assert(getCalledFunction(Call) == ExitTB);

    // Look for the last write to the PC
    auto &&[Result, NextPC] = PCH->getUniqueJumpTarget(Call->getParent());

    switch (Result) {
    case NextJumpTarget::Unique:
      // A constant store was born
      revng_assert(NextPC.isValid());
      pinConstantStoreInternal(NextPC, Call);
      break;

    case NextJumpTarget::Multiple:
      // Nothing to do, it's an indirect jump
      break;

    case NextJumpTarget::Helper:
      forceFallthroughAfterHelper(Call);
      break;

    default:
      revng_abort();
    }
  }

  return true;
}

static bool isPossiblyReturningHelper(ProgramCounterHandler *PCH,
                                      const MetaAddress &PC,
                                      llvm::Instruction *I) {
  // Is this a helper?
  auto *Call = getCallToHelper(I);
  if (Call == nullptr)
    return false;

  // Does this helper write something?
  auto UsedCSV = getCSVUsedByHelperCallIfAvailable(Call);
  if (not UsedCSV.has_value() or UsedCSV->Written.empty())
    return false;

  // Does this helper affect PC?
  auto AffectsPC = [PCH](GlobalVariable *CSV) { return PCH->affectsPC(CSV); };
  if (not llvm::any_of(UsedCSV->Written, AffectsPC))
    return false;

  // Obtain the name of the helper
  StringRef CalleeName;
  if (auto *Callee = getCalledFunction(Call))
    CalleeName = Callee->getName();

  // Returns the value of the n-th argument, if constant
  auto GetArgument = [Call](unsigned Index) -> std::optional<uint64_t> {
    Value *V = Call->getArgOperand(Index);
    auto *CI = dyn_cast_or_null<ConstantInt>(V);
    if (CI == nullptr)
      return std::nullopt;

    return CI->getLimitedValue();
  };

  // TODO: handle non-returning syscalls

  // Exclude architecture specific exceptions
  switch (PC.type()) {
  case MetaAddressType::Code_aarch64:
    // Handle calls to helper_exception_with_syndrome with EXCP_UDEF
    if (CalleeName == "helper_exception_with_syndrome"
        and GetArgument(1).value_or(0) == 1) {
      return false;
    }
    break;

  default:
    // TODO: add more
    break;
  }

  return true;
}

bool TDBP::forceFallthroughAfterHelper(CallInst *Call) {
  // If someone else already took care of the situation, quit
  if (getLimitedValue(Call->getArgOperand(0)) > 0)
    return false;

  const auto &[PC, Size] = JTM->getPC(Call);
  MetaAddress NextPC = PC + Size;

  bool ForceFallthrough = false;

  BasicBlock::reverse_iterator It(++Call->getReverseIterator());
  auto *BB = Call->getParent();
  auto EndIt = BB->rend();
  while (!ForceFallthrough) {
    while (It != EndIt) {
      Instruction *I = &*It;
      if (auto *Store = dyn_cast<StoreInst>(I)) {
        if (PCH->affectsPC(Store)) {
          // We found a PC-store, give up
          return false;
        }
      } else if (isPossiblyReturningHelper(PCH, PC, I)) {
        // We found a call to an helper
        ForceFallthrough = true;
        break;
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

  IRBuilder<> Builder(Call->getParent());
  Call->setArgOperand(0, Builder.getInt32(1));

  // Get the fallthrough basic block and emit a conditional branch, if not
  // possible simply jump to anyPC
  BasicBlock *AnyPC = JTM->anyPC();
  if (BasicBlock *NextPCBB = JTM->registerJT(NextPC, JTReason::PostHelper)) {
    PCH->buildHotPath(Builder, { NextPC, NextPCBB }, AnyPC);
  } else {
    Builder.CreateBr(AnyPC);
  }

  JTM->recordNewBranches(Call->getParent(), 1);

  return true;
}

bool TDBP::runOnModule(Module &M) {
  Function &F = *M.getFunction("root");
  pinConstantStore(F);
  pinMaterializedValues(F);
  return true;
}

MaterializedValue JumpTargetManager::readFromPointer(MetaAddress LoadAddress,
                                                     unsigned LoadSize,
                                                     bool IsLittleEndian) {
  auto NewAPInt = [LoadSize](uint64_t V) { return APInt(LoadSize * 8, V); };

  UnusedCodePointers.erase(LoadAddress);

  // Prevent overflow when computing the label interval
  MetaAddress EndAddress = LoadAddress + LoadSize;
  if (not EndAddress.isValid())
    return MaterializedValue::invalid();

  registerReadRange(LoadAddress, EndAddress);

  //
  // Check relocations
  //
  unsigned MatchCount = 0;
  MaterializedValue Result;

  // Check dynamic functions-related relocations
  for (const model::DynamicFunction &Function :
       Model->ImportedDynamicFunctions()) {
    for (const model::Relocation &Relocation : Function.Relocations()) {
      uint64_t Addend = Relocation.Addend();
      auto RelocationSize = model::RelocationType::getSize(Relocation.Type());
      if (LoadAddress == Relocation.Address() and LoadSize == RelocationSize) {
        // TODO: add this to model verify
        revng_assert(not StringRef(Function.Name()).contains('\0'));
        Result = MaterializedValue::fromSymbol(Function.Name(),
                                               NewAPInt(Addend));
        ++MatchCount;
      }
    }
  }

  // Check segment-related relocations
  for (const model::Segment &Segment : Model->Segments()) {
    for (const model::Relocation &Relocation : Segment.Relocations()) {
      uint64_t Addend = Relocation.Addend();
      auto RelocationSize = model::RelocationType::getSize(Relocation.Type());
      if (LoadAddress == Relocation.Address() and LoadSize == RelocationSize) {
        MetaAddress Address = Segment.StartAddress() + Addend;
        if (Address.isValid()) {
          Result = MaterializedValue::fromConstant(NewAPInt(Address.address()));
          ++MatchCount;
        } else {
          // TODO: log message
        }
      }
    }
  }

  if (MatchCount == 1) {
    return Result;
  } else if (MatchCount > 1) {
    // TODO: log message
  }

  // No labels found, fall back to read the raw value, if available
  auto MaybeValue = BinaryView.readInteger(LoadAddress,
                                           LoadSize,
                                           IsLittleEndian);

  if (MaybeValue)
    return MaterializedValue::fromConstant(NewAPInt(*MaybeValue));
  else
    return MaterializedValue::invalid();
}

JumpTargetManager::JumpTargetManager(Function *TheFunction,
                                     ProgramCounterHandler *PCH,
                                     CSAAFactory CreateCSAA,
                                     const TupleTree<model::Binary> &Model,
                                     const RawBinaryView &BinaryView) :
  TheModule(*TheFunction->getParent()),
  Context(TheModule.getContext()),
  TheFunction(TheFunction),
  OriginalInstructionAddresses(),
  JumpTargets(),
  ExitTB(nullptr),
  Dispatcher(nullptr),
  DispatcherSwitch(nullptr),
  CurrentCFGForm(CFGForm::UnknownForm),
  CreateCSAA(CreateCSAA),
  PCH(PCH),
  Model(Model),
  BinaryView(BinaryView) {

  FunctionType *ExitTBTy = FunctionType::get(Type::getVoidTy(Context),
                                             { Type::getInt32Ty(Context) },
                                             false);

  FunctionCallee ExitCallee = getOrInsertIRHelper("exitTB",
                                                  TheModule,
                                                  ExitTBTy);
  ExitTB = cast<Function>(ExitCallee.getCallee());
  FunctionTags::Marker.addTo(ExitTB);

  prepareDispatcher();

  // Collect executable ranges from the model
  ExecutableRanges = Model->executableRanges();

  if (RegisterJTLog.isEnabled()) {
    RegisterJTLog << "Executable ranges:\n";
    for (const auto &[Start, End] : ExecutableRanges) {
      RegisterJTLog << "  " << Start.toString() << "-" << End.toString()
                    << "\n";
    }
    RegisterJTLog << DoLog;
  }

  // Configure GlobalValueNumbering
  StringMap<cl::Option *> &Options(cl::getRegisteredOptions());
  getOption<bool>(Options, "enable-load-pre")->setInitialValue(false);
  getOption<unsigned>(Options, "memdep-block-scan-limit")->setInitialValue(100);
  // Increase the Cap of the clobbering calls (`getClobberingMemoryAccess()`) in
  // EarlyCSE, so MemorySSA is still useful in the Pass. This is needed to avoid
  // using of GVN Pass, which is very slow.
  const char *EarlyCSEOption = "earlycse-mssa-optimization-cap";
  getOption<unsigned>(Options, EarlyCSEOption)->setInitialValue(2000);

  // getOption<bool>(Options, "enable-pre")->setInitialValue(false);
  // getOption<uint32_t>(Options, "max-recurse-depth")->setInitialValue(10);
}

void JumpTargetManager::harvestGlobalData() {
  // Register ExtraCodeAddresses
  for (MetaAddress Address : Model->ExtraCodeAddresses())
    registerJT(Address, JTReason::GlobalData);

  for (auto &[Segment, Data] : BinaryView.segments()) {
    MetaAddress StartVirtualAddress = Segment.StartAddress();
    const unsigned char *DataStart = Data.begin();
    const unsigned char *DataEnd = Data.end();

    using namespace model::Architecture;
    bool IsLittleEndian = isLittleEndian(Model->Architecture());
    auto PointerSize = getPointerSize(Model->Architecture());
    using endianness = support::endianness;
    if (PointerSize == 8) {
      if (IsLittleEndian)
        findCodePointers<uint64_t, endianness::little>(StartVirtualAddress,
                                                       DataStart,
                                                       DataEnd);
      else
        findCodePointers<uint64_t, endianness::big>(StartVirtualAddress,
                                                    DataStart,
                                                    DataEnd);
    } else if (PointerSize == 4) {
      if (IsLittleEndian)
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
void JumpTargetManager::findCodePointers(MetaAddress StartVirtualAddress,
                                         const unsigned char *Start,
                                         const unsigned char *End) {
  using support::endianness;
  using support::endian::read;

  constexpr auto Step = sizeof(value_type);

  auto Cursor = Start;

  // Align the starting address: we want to scan one step at a time starting
  // from an aligned size
  auto Misalignment = StartVirtualAddress.address() % Step;
  if (Misalignment != 0)
    Cursor += Step - Misalignment;

  for (; Cursor < End - Step; Cursor += Step) {
    auto Read = read<value_type, static_cast<endianness>(endian), 1>;
    uint64_t RawValue = Read(Cursor);
    MetaAddress Value = fromPC(RawValue);
    if (Value.isInvalid())
      continue;

    BasicBlock *Result = registerJT(Value, JTReason::GlobalData);

    if (Result != nullptr)
      UnusedCodePointers.insert(StartVirtualAddress + (Cursor - Start));
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
BasicBlock *JumpTargetManager::newPC(MetaAddress PC, bool &ShouldContinue) {
  revng_assert(PC.isValid());

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
          revng_assert(ToPurge.contains(Result));
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
  if (OriginalInstructionAddresses.contains(PC)) {
    ShouldContinue = false;
    return registerJT(PC, JTReason::AmbiguousInstruction);
  }

  // We don't know anything about this PC
  return nullptr;
}

/// Save the PC-Instruction association for future use (jump target)
void JumpTargetManager::registerInstruction(MetaAddress PC,
                                            Instruction *Instruction) {
  revng_assert(PC.isValid());

  // Never save twice a PC
  revng_assert(!OriginalInstructionAddresses.contains(PC));
  OriginalInstructionAddresses[PC] = Instruction;
  revng_assert(Instruction->getParent() != nullptr);
}

// TODO: this is a candidate for BFSVisit
std::pair<MetaAddress, uint64_t>
JumpTargetManager::getPC(Instruction *TheInstruction) const {
  CallInst *NewPCCall = nullptr;
  llvm::DenseSet<BasicBlock *> Visited;
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
        auto *Callee = getCalledFunction(Marker);
        if (Callee != nullptr and Callee->getName() == "newpc") {

          // We found two distinct newpc leading to the requested instruction
          if (NewPCCall != nullptr)
            return { MetaAddress::invalid(), 0 };

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
        revng_assert(not(isPartOfRootDispatcher(Predecessor)));
      }

      for (BasicBlock *Predecessor : predecessors(BB)) {
        // Ignore already visited or empty BBs
        if (!Predecessor->empty() && !Visited.contains(Predecessor)) {
          WorkList.push(Predecessor->rbegin());
          Visited.insert(Predecessor);
        }
      }
    }
  }

  // Couldn't find the current PC
  if (NewPCCall == nullptr)
    return { MetaAddress::invalid(), 0 };

  using namespace NewPCArguments;
  MetaAddress PC = addressFromNewPC(NewPCCall);
  uint64_t Size = getLimitedValue(NewPCCall->getArgOperand(InstructionSize));
  revng_assert(Size != 0);
  return { PC, Size };
}

/// Class to iterate over all the BBs associated to a translated PC
class BasicBlockVisitor {
public:
  BasicBlockVisitor(const SwitchInst *Dispatcher) :
    Dispatcher(Dispatcher),
    JumpTargetIndex(0),
    JumpTargetsCount(Dispatcher->getNumSuccessors()),
    DL(Dispatcher->getParent()->getParent()->getParent()->getDataLayout()) {}

  void enqueue(BasicBlock *BB) {
    if (Visited.contains(BB))
      return;
    Visited.insert(BB);

    MetaAddress PC = getPC(BB);
    if (PC.isInvalid())
      SamePC.push(BB);
    else
      NewPC.push({ BB, PC });
  }

  std::pair<BasicBlock *, MetaAddress> pop() {
    if (!SamePC.empty()) {
      auto Result = SamePC.front();
      SamePC.pop();
      return { Result, MetaAddress::invalid() };
    } else if (!NewPC.empty()) {
      auto Result = NewPC.front();
      NewPC.pop();
      return Result;
    } else if (JumpTargetIndex < JumpTargetsCount) {
      BasicBlock *BB = Dispatcher->getSuccessor(JumpTargetIndex);
      JumpTargetIndex++;
      return { BB, getPC(BB) };
    } else {
      return { nullptr, MetaAddress::invalid() };
    }
  }

private:
  MetaAddress getPC(BasicBlock *BB) {
    if (!BB->empty()) {
      if (auto *Call = getCallTo(&*BB->begin(), "newpc"))
        return addressFromNewPC(Call);
    }

    return MetaAddress::invalid();
  }

private:
  const SwitchInst *Dispatcher = nullptr;
  unsigned JumpTargetIndex;
  unsigned JumpTargetsCount;
  const DataLayout &DL;
  llvm::DenseSet<BasicBlock *> Visited;
  std::queue<BasicBlock *> SamePC;
  std::queue<std::pair<BasicBlock *, MetaAddress>> NewPC;
};

void JumpTargetManager::fixPostHelperPC() {
  for (BasicBlock &BB : *TheFunction) {
    for (Instruction &I : BB) {
      if (auto *Call = getCallToHelper(&I)) {
        auto *Callee = cast<Function>(skipCasts(Call->getCalledOperand()));
        if (Callee->getName() == AbortFunctionName)
          continue;

        auto Written = std::move(getCSVUsedByHelperCall(Call).Written);
        auto WritesPC = [this](GlobalVariable *CSV) {
          return PCH->affectsPC(CSV);
        };
        auto End = Written.end();
        if (std::find_if(Written.begin(), End, WritesPC) != End) {
          IRBuilder<> Builder(Call->getParent(), ++Call->getIterator());
          PCH->deserializePC(Builder);
        }
      }
    }
  }
}

void JumpTargetManager::translateIndirectJumps() {
  if (ExitTB->use_empty())
    return;

  auto I = ExitTB->use_begin();
  while (I != ExitTB->use_end()) {
    Use &ExitTBUse = *I++;
    if (auto *Call = dyn_cast<CallInst>(ExitTBUse.getUser())) {
      if (getCalledFunction(Call) == ExitTB) {

        // Look for the last write to the PC
        BasicBlock *CallBB = Call->getParent();
        auto &&[Result, NextPC] = PCH->getUniqueJumpTarget(CallBB);

        if (NextPC.isValid() and isExecutableAddress(NextPC)) {
          revng_check(Result != NextJumpTarget::Unique
                      and "Direct jumps should not be handled here");
        }

        if (getLimitedValue(Call->getArgOperand(0)) == 0) {
          exitTBCleanup(Call);
          BranchInst::Create(AnyPC, Call);
        }

        eraseFromParent(Call);
      }
    }
  }

  revng_assert(ExitTB->use_empty());
  eraseFromParent(ExitTB);
  ExitTB = nullptr;
}

JumpTargetManager::BlockWithAddress JumpTargetManager::peek() {
  // If we just harvested new branches, keep exploring
  do {
    harvest();
  } while (Unexplored.empty() and NewBranches != 0);

  // Purge all the partial translations we know might be wrong
  for (BasicBlock *BB : ToPurge)
    purgeTranslation(BB);
  ToPurge.clear();

  // Puring leaves some unreachable blocks behind: collect them
  EliminateUnreachableBlocks(*TheFunction);

  if (Unexplored.empty()) {
    revng_log(JTCountLog, "We're done looking for jump targets");
    return NoMoreTargets;
  } else {
    BlockWithAddress Result = Unexplored.back();
    Unexplored.pop_back();
    return Result;
  }
}

BasicBlock *JumpTargetManager::getBlockAt(MetaAddress PC) {
  revng_assert(PC.isValid());

  auto TargetIt = JumpTargets.find(PC);
  revng_assert(TargetIt != JumpTargets.end());
  return TargetIt->second.head();
}

/// Check if among \p BB's predecessors there's \p Target
inline bool hasRootDispatcherPredecessor(llvm::BasicBlock *BB) {
  for (llvm::BasicBlock *Predecessor : predecessors(BB))
    if (isPartOfRootDispatcher(Predecessor))
      return true;
  return false;
}

void JumpTargetManager::purgeTranslation(BasicBlock *Start) {
  OnceQueue<BasicBlock *> Queue;
  Queue.insert(Start);

  revng_log(RegisterJTLog,
            "Purging " << getName(Start) << " so it can be translated again");

  // Collect all the descendants, except if we meet a jump target
  while (!Queue.empty()) {
    BasicBlock *BB = Queue.pop();
    Instruction *Terminator = BB->getTerminator();
    for (BasicBlock *Successor : successors(Terminator)) {
      if (isTranslatedBB(Successor) and not isJumpTarget(Successor)
          and not hasRootDispatcherPredecessor(Successor)) {
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
    while (!BB->empty()) {
      Instruction *I = &*(--BB->end());

      if (CallInst *Call = getCallTo(I, "newpc")) {
        OriginalInstructionAddresses.erase(addressFromNewPC(Call));
      }
      eraseInstruction(I);
    }
  }

  // Remove Start, since we want to keep it (even if empty)
  Visited.erase(Start);

  for (BasicBlock *BB : Visited)
    eraseFromParent(BB);
}

// TODO: register Reason
BasicBlock *JumpTargetManager::registerJT(MetaAddress PC,
                                          JTReason::Values Reason) {
  revng_check(PC.isValid());

  if (not isPC(PC))
    return nullptr;

  revng_log(RegisterJTLog,
            "Registering bb." << nameForAddress(PC) << " for "
                              << JTReason::getName(Reason));
  LoggerIndent<> Indent(RegisterJTLog);

  // Do we already have a BasicBlock for this PC?
  BlockMap::iterator TargetIt = JumpTargets.find(PC);
  if (TargetIt != JumpTargets.end()) {
    // Case 1: there's already a BasicBlock for that address, return it
    revng_log(RegisterJTLog, "We already translated this block");
    BasicBlock *BB = TargetIt->second.head();
    TargetIt->second.setReason(Reason);
    return BB;
  }

  // Did we already meet this PC (i.e. do we know what's the associated
  // instruction)?
  BasicBlock *NewBlock = nullptr;
  InstructionMap::iterator InstrIt = OriginalInstructionAddresses.find(PC);
  if (InstrIt != OriginalInstructionAddresses.end()) {
    revng_log(RegisterJTLog,
              "We already met this PC, but not as a jump target");
    // Case 2: the address has already been met, but needs to be promoted to
    //         BasicBlock level.
    Instruction *I = InstrIt->second;
    BasicBlock *ContainingBlock = I->getParent();
    if (isFirst(I)) {
      NewBlock = ContainingBlock;
    } else {
      revng_log(RegisterJTLog, "Splitting the basic block it was in");
      revng_assert(I != nullptr && I->getIterator() != ContainingBlock->end());
      NewBlock = ContainingBlock->splitBasicBlock(I);
    }

    // Register the basic block and all of its descendants to be purged so that
    // we can retranslate this PC
    // TODO: this might create a problem if QEMU generates control flow that
    //       crosses an instruction boundary
    revng_log(RegisterJTLog, "Registering the block for re-translation");
    ToPurge.insert(NewBlock);

  } else {
    // Case 3: the address has never been met, create a temporary one, register
    // it for future exploration and return it
    revng_log(RegisterJTLog, "Registering the block for translation");
    NewBlock = BasicBlock::Create(Context, "", TheFunction);
  }

  Unexplored.push_back(BlockWithAddress(PC, NewBlock));

  std::stringstream Name;
  Name << "bb." << nameForAddress(PC);
  NewBlock->setName(Name.str());

  // Create a case for the address associated to the new block, if the
  // dispatcher has already been emitted
  if (DispatcherSwitch != nullptr) {
    PCH->addCaseToDispatcher(DispatcherSwitch,
                             { PC, NewBlock },
                             BlockType::RootDispatcherHelperBlock);
  }

  // Associate the PC with the chosen basic block
  auto &NewJumpTarget = JumpTargets[PC];
  NewJumpTarget = JumpTarget(NewBlock, Reason);

  if (AftedAddingFunctionEntries)
    NewJumpTarget.setReason(JTReason::DependsOnModelFunction);

  // PC was not a jump target, record it as new
  ValueMaterializerPCWhiteList.insert(PC);

  return NewBlock;
}

void JumpTargetManager::registerReadRange(MetaAddress StartAddress,
                                          MetaAddress EndAddress) {
  if (not isMapped(StartAddress, EndAddress))
    return;

  using interval = boost::icl::interval<MetaAddress, CompareAddress>;
  ReadIntervalSet += interval::right_open(StartAddress, EndAddress);
}

void JumpTargetManager::prepareDispatcher() {
  IRBuilder<> Builder(Context);
  QuickMetadata QMD(Context);

  // Create the first block of the dispatcher
  BasicBlock *Entry = BasicBlock::Create(Context,
                                         "dispatcher.entry",
                                         TheFunction);

  // The default case of the switch statement it's an unhandled cases
  DispatcherFail = BasicBlock::Create(Context,
                                      "dispatcher.default",
                                      TheFunction);
  Builder.SetInsertPoint(DispatcherFail);

  FunctionCallee UnknownPC = TheModule.getFunction("unknown_pc");
  {
    auto *UnknownPCFunction = cast<Function>(skipCasts(UnknownPC.getCallee()));
    FunctionTags::Exceptional.addTo(UnknownPCFunction);
  }

  PCH->setCurrentPCPlainMetaAddress(Builder);

  Builder.CreateCall(UnknownPC);

  auto *FailUnreachable = Builder.CreateUnreachable();
  setBlockType(FailUnreachable, BlockType::DispatcherFailureBlock);

  Dispatcher = Entry;

  // Create basic blocks to handle jumps to any PC and to a PC we didn't expect
  AnyPC = BasicBlock::Create(Context, "anypc", TheFunction);
  UnexpectedPC = BasicBlock::Create(Context, "unexpectedpc", TheFunction);
}

static void purge(BasicBlock *BB) {
  // Allow up to a single instruction in the basic block
  if (!BB->empty())
    eraseFromParent(&*BB->begin());
  revng_assert(BB->empty());
}

llvm::DenseSet<BasicBlock *> JumpTargetManager::computeUnreachable() const {
  ReversePostOrderTraversal<BasicBlock *> RPOT(&TheFunction->getEntryBlock());
  llvm::DenseSet<BasicBlock *> Reachable;
  for (BasicBlock *BB : RPOT)
    Reachable.insert(BB);

  // TODO: why is isTranslatedBB(&BB) necessary?
  llvm::DenseSet<BasicBlock *> Unreachable;
  for (BasicBlock &BB : *TheFunction)
    if (not Reachable.contains(&BB) and isTranslatedBB(&BB))
      Unreachable.insert(&BB);

  return Unreachable;
}

void JumpTargetManager::setCFGForm(CFGForm::Values NewForm,
                                   MetaAddressSet *JumpTargetsWhitelist) {
  revng_assert(CurrentCFGForm != NewForm);
  revng_assert(NewForm != CFGForm::UnknownForm);

  CFGForm::Values OldForm = CurrentCFGForm;
  CurrentCFGForm = NewForm;

  //
  // Recreate AnyPC and UnexpectedPC
  //
  switch (NewForm) {
  case CFGForm::SemanticPreserving:
    purge(AnyPC);
    BranchInst::Create(dispatcher(), AnyPC);
    // TODO: Here we should have an hard fail, since it's the situation in
    //       which we expected to know where execution could go but we made a
    //       mistake.
    purge(UnexpectedPC);
    BranchInst::Create(dispatcher(), UnexpectedPC);
    break;

  case CFGForm::RecoveredOnly:
  case CFGForm::NoFunctionCalls:
    purge(AnyPC);
    new UnreachableInst(Context, AnyPC);
    purge(UnexpectedPC);
    new UnreachableInst(Context, UnexpectedPC);
    break;

  default:
    revng_abort("Not implemented yet");
  }

  setBlockType(AnyPC->getTerminator(), BlockType::AnyPCBlock);
  setBlockType(UnexpectedPC->getTerminator(), BlockType::UnexpectedPCBlock);

  // Adjust successors of jump instructions tagged as function calls:
  //
  // * NoFunctionsCallsCFG: the successor is the fallthrough
  // * otherwise: the successor is the callee
  if (NewForm == CFGForm::NoFunctionCalls
      || OldForm == CFGForm::NoFunctionCalls) {
    if (auto *FunctionCall = getIRHelper("function_call", TheModule)) {
      for (User *U : FunctionCall->users()) {
        auto *Call = cast<CallInst>(U);

        // Ignore indirect calls
        // TODO: why this is needed is unclear
        if (isa<ConstantPointerNull>(Call->getArgOperand(0)))
          continue;

        Instruction *Terminator = nextNonMarker(Call);
        revng_assert(Terminator->getNumSuccessors() == 1);

        // Get the correct argument, the first is the callee, the second the
        // return basic block
        int OperandIndex = NewForm == CFGForm::NoFunctionCalls ? 1 : 0;
        Value *Op = Call->getArgOperand(OperandIndex);
        BasicBlock *NewSuccessor = cast<BlockAddress>(Op)->getBasicBlock();
        Terminator->setSuccessor(0, NewSuccessor);
      }
    }
  }

  rebuildDispatcher(JumpTargetsWhitelist);
}

void JumpTargetManager::rebuildDispatcher(MetaAddressSet *Whitelist) {

  if (DispatcherSwitch != nullptr) {
    revng_assert(DispatcherSwitch->getParent() == Dispatcher);

    // Purge the old dispatcher
    PCH->destroyDispatcher(DispatcherSwitch);
  }

  ProgramCounterHandler::DispatcherTargets Targets;

  // Add all the (whitelisted) jump targets if we're using the
  // SemanticPreserving, or only those with no predecessors.
  bool IsWhitelistActive = (Whitelist != nullptr);
  for (auto &[PC, JumpTarget] : JumpTargets) {
    BasicBlock *BB = JumpTarget.head();
    bool IsWhitelisted = (not IsWhitelistActive or Whitelist->contains(PC));
    if ((CurrentCFGForm == CFGForm::SemanticPreserving
         or not hasPredecessors(BB))
        and IsWhitelisted) {
      Targets.emplace_back(PC, BB);
    }
  }

  constexpr auto RDHB = BlockType::RootDispatcherHelperBlock;
  const auto &DispatcherInfo = PCH->buildDispatcher(Targets,
                                                    Dispatcher,
                                                    DispatcherFail,
                                                    RDHB);
  DispatcherSwitch = DispatcherInfo.Switch;

  // The switch is the terminator of the dispatcher basic block
  setBlockType(DispatcherSwitch, BlockType::RootDispatcherBlock);

  //
  // Make sure every generated basic block is reachable
  //
  if (CurrentCFGForm != CFGForm::SemanticPreserving) {
    llvm::DenseSet<BasicBlock *> Reachable;
    // Compute the set of jump targets currently reachables from the dispatcher
    for (BasicBlock *DFSBB : llvm::depth_first(DispatcherSwitch->getParent())) {
      Reachable.insert(DFSBB);
    }

    // Identify all the unreachable jump targets, and add an edge from the
    // dispatcher to them. Note that the order we use to iterate over
    // `JumpTargets` is fundamental, because we want to connect to the
    // dispatcher first the jump target with the lower program counter. At the
    // same time, we will mark as reachable all the jump targets that are
    // transitively reachable from the elected jump target. In this way, we
    // connect to the dispatcher all the blocks belonging to a separate SCC that
    // were not reachable initially (e.g., a function only indirectly called).
    for (const auto &[PC, JT] : JumpTargets) {
      BasicBlock *BB = JT.head();
      bool IsWhitelisted = (not IsWhitelistActive or Whitelist->contains(PC));

      // Add to the switch all the unreachable jump targets whose reason is not
      // just direct jump
      if (not Reachable.contains(BB) and IsWhitelisted
          and not JT.isOnlyReason(JTReason::DirectJump,
                                  JTReason::DependsOnModelFunction)) {
        PCH->addCaseToDispatcher(DispatcherSwitch,
                                 { PC, BB },
                                 BlockType::RootDispatcherHelperBlock);

        // Add to the `Reachable` set also all the jump targets that are now
        // reachable. We do this with a with a simple DFS visit from the
        // newly connected one.
        for (BasicBlock *DFSBB : llvm::depth_first(BB)) {
          Reachable.insert(DFSBB);
        }
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

CallInst *JumpTargetManager::getJumpTarget(BasicBlock *Target) {
  for (BasicBlock *BB : inverse_depth_first(Target)) {
    if (auto *Call = dyn_cast<CallInst>(&*BB->begin())) {
      auto MA = addressFromNewPC(Call);
      if (MA.isValid() and isJumpTarget(MA))
        return Call;
    }
  }

  return nullptr;
}

// Harvesting proceeds trying to avoid to run expensive analyses if not strictly
// necessary. To do this we keep in mind two aspects: do we have new basic
// blocks to visit? If so, we avoid any further anyalysis and give back control
// to the translator. If not, we proceed with other analyses until we either
// find a new basic block to translate. If we can't find a new block to
// translate we proceed as long as we are able to create new edges on the CFG
// (not considering the dispatcher).
void JumpTargetManager::harvest() {
  Task T(11, "Harvesting");
  HarvestingStats.push("harvest 0");

  if (empty()) {
    T.advance("Simple literals");
    HarvestingStats.push("harvest 1: SimpleLiterals");
    revng_log(JTCountLog, "Collecting simple literals");
    for (MetaAddress PC : SimpleLiterals)
      registerJT(PC, JTReason::SimpleLiteral);
    SimpleLiterals.clear();
  }

  if (empty()) {
    T.advance("SROA + InstSimplify + TBDP");
    HarvestingStats.push("harvest 2: SROA + InstSimplify + TBDP");

    // Safely erase all unreachable blocks
    llvm::DenseSet<BasicBlock *> Unreachable = computeUnreachable();
    for (BasicBlock *BB : Unreachable)
      BB->dropAllReferences();
    for (BasicBlock *BB : Unreachable)
      eraseFromParent(BB);

    // TODO: move me to a commit function

    // Update the third argument of newpc calls (isJT, i.e., is this instruction
    // a jump target?)
    IRBuilder<> Builder(Context);
    Function *NewPCFunction = getIRHelper("newpc", TheModule);
    if (NewPCFunction != nullptr) {
      for (User *U : NewPCFunction->users()) {
        auto *Call = cast<CallInst>(U);
        if (Call->getParent() != nullptr) {
          // Report the instruction on
          // the coverage CSV
          MetaAddress PC = addressFromNewPC(Call);
          bool IsJT = isJumpTarget(PC);
          Call->setArgOperand(2, Builder.getInt32(static_cast<uint32_t>(IsJT)));
        }
      }
    }

    revng::verify(&TheModule);

    revng_log(JTCountLog, "Preliminary harvesting");

    HarvestingStats.push("InstSimplify");
    legacy::FunctionPassManager OptimizingPM(&TheModule);
    OptimizingPM.add(createSROAPass());
    OptimizingPM.add(createInstSimplifyLegacyPass());
    OptimizingPM.doInitialization();
    OptimizingPM.run(*TheFunction);
    OptimizingPM.doFinalization();

    legacy::PassManager PreliminaryBranchesPM;
    PreliminaryBranchesPM.add(new TranslateDirectBranchesPass(this));
    PreliminaryBranchesPM.run(TheModule);

    if (empty()) {
      T.advance("InstructionCombining + TBDP");
      HarvestingStats.push("harvest 3: InstructionCombining + TBDP");

      legacy::FunctionPassManager OptimizingPM(&TheModule);
      OptimizingPM.add(createInstructionCombiningPass(1));
      OptimizingPM.doInitialization();
      OptimizingPM.run(*TheFunction);
      OptimizingPM.doFinalization();

      legacy::PassManager PreliminaryBranchesPM;
      PreliminaryBranchesPM.add(new TranslateDirectBranchesPass(this));
      PreliminaryBranchesPM.run(TheModule);
    }

    if (empty()) {
      T.advance("Advanced Value Info");
      HarvestingStats.push("harvest 4: cloneOptimizeAndHarvest");
      revng_log(JTCountLog, "Harvesting with Advanced Value Info");
      RootAnalyzer(*this).cloneOptimizeAndHarvest(TheFunction);
    }

    if (empty()) {
      // Register model::Function entry nodes

      AftedAddingFunctionEntries = true;

      DisableTracking Guard(*Model);
      for (const model::Function &Function : Model->Functions())
        registerJT(Function.Entry(), JTReason::FunctionSymbol);
    }

    // TODO: eventually, `setCFGForm` should be replaced by using a CustomCFG
    // To improve the quality of our analysis, keep in the CFG only the edges we
    // where able to recover (e.g., no jumps to the dispatcher)
    setCFGForm(CFGForm::RecoveredOnly);

    NewBranches = 0;
    legacy::PassManager AnalysisPM;
    AnalysisPM.add(new TranslateDirectBranchesPass(this));
    AnalysisPM.run(TheModule);

    // Restore the CFG
    setCFGForm(CFGForm::SemanticPreserving);

    if (JTCountLog.isEnabled()) {
      JTCountLog << std::dec << Unexplored.size() << " new jump targets and "
                 << NewBranches << " new branches were found" << DoLog;
    }
  }
}

using BWA = JumpTargetManager::BlockWithAddress;
using JTM = JumpTargetManager;
const BWA JTM::NoMoreTargets = BWA(MetaAddress::invalid(), nullptr);
