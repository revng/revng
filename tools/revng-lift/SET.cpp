/// \file SET.cpp
///
/// \brief Simple Expression Tracker pass implementation
///
/// This file is composed by three main parts: the OperationsStack
/// implementation, the SET algorithm and the SET pass

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <iterator>

// LLVM includes
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"

// Local libraries includes
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/revng.h"

// Local includes
#include "JumpTargetManager.h"
#include "OSRA.h"
#include "SET.h"

using namespace llvm;
using std::make_pair;

static Logger<> NewEdgesLog("new-edges");

class MaterializedValue {
private:
  bool IsValid;
  Optional<StringRef> SymbolName;
  uint64_t Value;

private:
  MaterializedValue() : IsValid(false), Value(0) {}

public:
  MaterializedValue(uint64_t Value) : IsValid(true), Value(Value) {}
  MaterializedValue(StringRef Name, uint64_t Offset) :
    IsValid(true),
    SymbolName(Name),
    Value(Offset) {}

public:
  static MaterializedValue invalid() { return MaterializedValue(); }

public:
  uint64_t value() const {
    revng_assert(isValid());
    return Value;
  }
  bool isValid() const { return IsValid; }
  bool hasSymbol() const { return SymbolName.hasValue(); }
  StringRef symbolName() const {
    revng_assert(isValid());
    revng_assert(hasSymbol());
    return *SymbolName;
  }
};

/// \brief Stack to keep track of the operations generating a specific value
///
/// The OperationsStacks offers the following features:
///
/// * it doesn't insert more than once an item (to avoid loops)
/// * it can traverse the stack from top to bottom to produce a value and, if
///   required, register it with the JumpTargetManager
/// * cut the stack to a certain height
/// * manage the lifetime of orphan instructions it contains
/// * keep track of all the possible values assumed since the last reset and
///   whether this information is precise or not
class OperationsStack {
public:
  OperationsStack(JumpTargetManager *JTM,
                  const DataLayout &DL,
                  FunctionCallIdentification *FCI) :
    JTM(JTM),
    DL(DL),
    LoadsCount(0),
    FCI(FCI) {
    reset();
  }

  ~OperationsStack() { reset(); }

  void explore(Constant *NewOperand);
  uint64_t materializeSimple(Constant *NewOperand) {
    MaterializedValue Result = materialize(NewOperand, false);
    if (not Result.isValid())
      return 0;

    revng_assert(not Result.hasSymbol());

    return Result.value();
  }
  MaterializedValue materialize(Constant *NewOperand, bool HandleSymbols);

  /// \brief What values should be tracked
  enum TrackingType {
    None, ///< Don't track anything
    PCsOnly, ///< Track only values which can be PCs
  };

  /// \brief Clean the operations stack
  void reset() {
    // Delete all the temporary instructions we created
    for (Instruction *I : Operations) {
      if (I->getParent() == nullptr) {
        I->dropUnknownNonDebugMetadata();
        I->deleteValue();
      }
    }

    Operations.clear();
    OperationsSet.clear();
    TrackedValues.clear();
    Approximate = false;
    Tracking = None;
    IsPCStore = false;
    SetsSyscallNumber = false;
    Target = nullptr;
    LoadsCount = 0;
  }

  void reset(StoreInst *Store) {
    reset();
    IsPCStore = JTM->isPCReg(Store->getPointerOperand());
    if (IsPCStore)
      Tracking = OperationsStack::PCsOnly;
    SetsSyscallNumber = JTM->noReturn().setsSyscallNumber(Store);
    Target = Store;
  }

  void registerPCs() const {
    const auto SETToPC = JTReason::SETToPC;
    const auto SETNotToPC = JTReason::SETNotToPC;
    for (auto &P : NewPCs) {

      if (not JTM->hasJT(P.first) and NewEdgesLog.isEnabled()) {
        uint64_t Source = JTM->getPC(Target).first;
        uint64_t Destination = P.first;
        NewEdgesLog << std::hex << "0x" << Source << " -> 0x" << Destination
                    << " (" << getName(Target->getParent()) << " -> "
                    << JTM->nameForAddress(Destination) << ")" << DoLog;
      }

      JTM->registerJT(P.first, P.second ? SETToPC : SETNotToPC);
    }
  }

  void registerLoadAddresses() const {
    for (uint64_t Address : LoadAddresses)
      JTM->markJT(Address, JTReason::LoadAddress);
  }

  void cut(unsigned Height) {
    revng_assert(Height <= Operations.size());
    while (Height != Operations.size()) {
      Instruction *Op = Operations.back();
      auto It = OperationsSet.find(Op);
      if (It != OperationsSet.end()) {
        OperationsSet.erase(It);
      } else if (isa<BinaryOperator>(Op)) {
        // It's not in OperationsSet, it might a binary instruction where we
        // forced one operand to be constant, or an instruction generated from a
        // constant unary expression
        unsigned FreeOpIndex = isa<Constant>(Op->getOperand(0)) ? 1 : 0;
        auto *FreeOp = cast<Instruction>(Op->getOperand(FreeOpIndex));
        auto It = OperationsSet.find(FreeOp);
        revng_assert(It != OperationsSet.end());
        OperationsSet.erase(It);
      }

      if (isa<LoadInst>(Op)) {
        revng_assert(LoadsCount > 0);
        LoadsCount--;
      }

      // We have the ownership of instruction without parent
      if (Op->getParent() == nullptr) {
        Op->dropUnknownNonDebugMetadata();
        Op->deleteValue();
      }

      Operations.pop_back();
    }
  }

  bool insertIfNew(Instruction *I) { return insertIfNew(I, I); }

  bool insertIfNew(Instruction *I, Instruction *Ref) {
    if (OperationsSet.find(Ref) == OperationsSet.end()) {
      insert(I);
      OperationsSet.insert(Ref);
      return true;
    }

    // If the given instruction doesn't have a parent we take ownership of it
    if (I->getParent() == nullptr) {
      I->dropUnknownNonDebugMetadata();
      I->deleteValue();
    }

    return false;
  }

  void insert(Instruction *I) {
    if (isa<LoadInst>(I))
      LoadsCount++;

    Operations.push_back(I);
  }

  void setApproximate() { Approximate = true; }

  bool isApproximate() const { return Approximate; }

  unsigned height() const { return Operations.size(); }
  bool empty() const { return Operations.empty(); }

  /// \brief Get the type of the free operand of the topmost stack element
  Type *topType() const {
    Type *Result = nullptr;
    bool NonConstFound = false;

    for (Value *Op : Operations.back()->operand_values()) {
      if (!isa<Constant>(Op)) {
        revng_assert(!NonConstFound);
        NonConstFound = true;
        Result = Op->getType();
      }
    }

    revng_assert(Result != nullptr);
    return Result;
  }

  std::vector<uint64_t> trackedValues() const {
    std::vector<uint64_t> Result;
    Result.reserve(TrackedValues.size());
    std::copy(TrackedValues.begin(),
              TrackedValues.end(),
              std::back_inserter(Result));
    return Result;
  }

  bool hasTrackedValues() const { return not TrackedValues.empty(); }

  bool readsMemory() const { return LoadsCount > 0; }

  void dump() const debug_function { dump(dbg); }

  template<typename O>
  void dump(O &Output) const {
    for (Instruction *I : Operations)
      Output << dumpToString(I) << "\n";
  }

private:
  JumpTargetManager *JTM;
  const DataLayout &DL;

  std::vector<Instruction *> Operations;
  std::set<Instruction *> OperationsSet;
  std::set<std::pair<uint64_t, bool>> NewPCs;
  std::set<uint64_t> TrackedValues;
  std::set<uint64_t> LoadAddresses;

  bool Approximate;
  TrackingType Tracking;
  bool IsPCStore;
  bool SetsSyscallNumber;
  unsigned LoadsCount;

  Instruction *Target;
  FunctionCallIdentification *FCI;

  /// Calls to `function_call` targeting multiple different dynamic symbols
  std::set<CallInst *> MultipleTargetsCalls;
};

MaterializedValue
OperationsStack::materialize(Constant *NewOperand, bool HandleSymbols) {
  Optional<StringRef> SymbolName;

  for (Instruction *I : make_range(Operations.rbegin(), Operations.rend())) {
    if (auto *Load = dyn_cast<LoadInst>(I)) {
      // OK, we've got a load, let's see if the load address is constant
      revng_assert(NewOperand != nullptr && !isa<UndefValue>(NewOperand));

      if (SymbolName)
        return MaterializedValue::invalid();

      uint64_t LoadAddress = getZExtValue(NewOperand, DL);
      unsigned LoadSize;

      // Read the value using the endianess of the destination architecture,
      // since, if there's a mismatch, in the stack we will also have a byteswap
      // instruction
      using Endianess = BinaryFile::Endianess;
      Endianess E = (DL.isLittleEndian() ? Endianess::LittleEndian :
                                           Endianess::BigEndian);
      if (Load->getType()->isIntegerTy()) {
        LoadSize = Load->getType()->getPrimitiveSizeInBits() / 8;
        revng_assert(LoadSize != 0);
        NewOperand = JTM->readConstantInt(NewOperand, LoadSize, E);
      } else if (Load->getType()->isPointerTy()) {
        LoadSize = JTM->binary().architecture().pointerSize() / 8;
        NewOperand = JTM->readConstantPointer(NewOperand, Load->getType(), E);
      } else {
        revng_abort();
      }

      // Prevent overflow when computing the label interval
      if (LoadAddress + LoadSize < LoadAddress) {
        return MaterializedValue::invalid();
      }

      const auto &Labels = JTM->binary().labels();
      using interval = boost::icl::interval<uint64_t>;
      auto Interval = interval::right_open(LoadAddress, LoadAddress + LoadSize);
      auto It = Labels.find(Interval);
      if (It != Labels.end()) {
        const Label *Match = nullptr;
        for (const Label *Candidate : It->second) {
          if (Candidate->size() == LoadSize
              and (Candidate->isAbsoluteValue()
                   or Candidate->isBaseRelativeValue()
                   or (HandleSymbols and Candidate->isSymbolRelativeValue()))) {
            revng_assert(Match == nullptr,
                         "Multiple value labels at the same location");
            Match = Candidate;
          }
        }

        if (Match != nullptr) {
          uint64_t Value;
          switch (Match->type()) {
          case LabelType::AbsoluteValue:
            Value = Match->value();
            break;

          case LabelType::BaseRelativeValue:
            Value = JTM->binary().relocate(Match->value());
            break;

          case LabelType::SymbolRelativeValue:
            Value = Match->offset();
            SymbolName = Match->symbolName();
            break;

          default:
            revng_abort();
          }

          NewOperand = ConstantInt::get(Load->getType(), Value);
        }
      }

      if (NewOperand == nullptr)
        break;

    } else if (auto *Call = dyn_cast<CallInst>(I)) {
      Function *Callee = Call->getCalledFunction();
      revng_assert(Callee != nullptr
                   && Callee->getIntrinsicID() == Intrinsic::bswap);

      uint64_t Value = NewOperand->getUniqueInteger().getLimitedValue();

      Type *T = NewOperand->getType();
      if (T->isIntegerTy(16))
        Value = ByteSwap_16(Value);
      else if (T->isIntegerTy(32))
        Value = ByteSwap_32(Value);
      else if (T->isIntegerTy(64))
        Value = ByteSwap_64(Value);
      else
        revng_unreachable("Unexpected type");

      NewOperand = ConstantInt::get(T, Value);
    } else {

      if (SymbolName) {
        // In case the result is relative to a symbol, whitelist the allowed
        // instructions
        switch (I->getOpcode()) {
        case Instruction::Add:
        case Instruction::Sub:
        case Instruction::And:
          break;
        default:
          if (I->getNumOperands() > 1)
            return MaterializedValue::invalid();
        }
      }

      // Replace non-const operand with NewOperand
      std::vector<Constant *> Operands;
      bool NonConstFound = false;

      for (Value *Op : I->operand_values()) {
        if (auto *Const = dyn_cast<Constant>(Op)) {
          Operands.push_back(Const);
        } else {
          revng_assert(!NonConstFound);
          NonConstFound = true;
          NewOperand = ConstantExpr::getTruncOrBitCast(NewOperand,
                                                       Op->getType());
          Operands.push_back(NewOperand);
        }
      }

      NewOperand = ConstantFoldInstOperands(I, Operands, DL);
      revng_assert(NewOperand != nullptr);
      // TODO: this is an hack hiding a bigger problem
      if (isa<UndefValue>(NewOperand)) {
        NewOperand = nullptr;
        break;
      }
    }
  }

  // We made it, mark the value to be explored
  if (NewOperand != nullptr) {
    revng_assert(!isa<UndefValue>(NewOperand));
    uint64_t Value = getZExtValue(NewOperand, DL);
    if (SymbolName)
      return MaterializedValue(*SymbolName, Value);
    else
      return MaterializedValue(Value);
  }

  return MaterializedValue::invalid();
}

void OperationsStack::explore(Constant *NewOperand) {
  MaterializedValue SymbolicValue = materialize(NewOperand, true);

  if (not SymbolicValue.isValid())
    return;

  if (SymbolicValue.hasSymbol()) {
    if (IsPCStore and SymbolicValue.value() == 0) {
      if (CallInst *Call = FCI->getCall(Target->getParent())) {

        if (MultipleTargetsCalls.count(Call) != 0)
          return;

        LLVMContext &Context = getContext(Call);
        QuickMetadata QMD(Context);
        auto *Callee = cast<Constant>(Call->getOperand(0));
        revng_assert(Callee->isNullValue(), "Direct call to external symbol");

        StringRef Name = SymbolicValue.symbolName();

        Value *Old = Call->getOperand(4);
        if (not isa<ConstantPointerNull>(Old)) {
          auto *Casted = cast<ConstantExpr>(Old)->getOperand(0);
          auto *Initializer = cast<GlobalVariable>(Casted)->getInitializer();
          auto String = cast<ConstantDataArray>(Initializer)->getAsString();

          // If this call can target multiple external symbols, simply don't tag
          // it
          // TODO: we should duplicate the call
          if (String.drop_back() != Name) {
            MultipleTargetsCalls.insert(Call);
            auto *PointerTy = dyn_cast<PointerType>(Old->getType());
            Call->setOperand(4, ConstantPointerNull::get(PointerTy));
            return;
          }
        }

        Module *M = Call->getParent()->getParent()->getParent();
        Constant *String = getUniqueString(M,
                                           "revng.input.symbol-names",
                                           Name,
                                           Twine("symbol_") + Name);
        Call->setOperand(4, String);
      }
    }

    return;
  }

  uint64_t Value = SymbolicValue.value();

  bool IsStore = Target != nullptr;
  revng_assert(!(!IsStore && (Tracking == PCsOnly || SetsSyscallNumber)));

  if (IsStore) {
    if (Value != 0 && JTM->isPC(Value))
      NewPCs.insert({ Value, IsPCStore });

    if (Value != 0 && (Tracking == PCsOnly && JTM->isPC(Value)))
      TrackedValues.insert(Value);

    if (SetsSyscallNumber) {
      Instruction *Top = Operations.size() == 0 ? Target : Operations.back();

      // TODO: don't ignore temporary instructions
      if (Top->getParent() != nullptr)
        JTM->noReturn().registerKiller(Value, Top, Target);
    }
  } else {
    // It's a load
    LoadAddresses.insert(Value);
  }
}

/// \brief Simple Expression Tracker implementation
class SET {
public:
  SET(Function &F,
      JumpTargetManager *JTM,
      OSRAPass *OSRA,
      FunctionCallIdentification *FCI,
      std::set<BasicBlock *> *Visited,
      std::vector<SETPass::JumpInfo> &Jumps) :
    DL(F.getParent()->getDataLayout()),
    JTM(JTM),
    OS(JTM, DL, FCI),
    F(F),
    OSRA(OSRA),
    Visited(Visited),
    Jumps(Jumps) {}

  /// \brief Run the Simple Expression Tracker on F
  bool run();

private:
  /// \brief Enqueue all the store seen by the Start load instruction
  /// \return true if it was possible to fully handle all instruction writing to
  ///         the source of the load instruction.
  bool enqueueStores(LoadInst *Load);

  /// \brief Process \p V
  /// \return a boolean indicating whether V has been handled properly and a new
  ///         Value from which SET should proceed
  Value *handleInstruction(Instruction *Target, Value *V);

  /// \brief Process \p V using information from OSRA
  /// \return true if the instruction was handled.
  bool handleInstructionWithOSRA(Instruction *Target, Value *V);

  void collectMetadata();

private:
  const unsigned MaxDepth = 3;
  const DataLayout &DL;
  JumpTargetManager *JTM;
  OperationsStack OS;
  Function &F;
  OSRAPass *OSRA;
  std::set<BasicBlock *> *Visited;
  std::vector<std::pair<Value *, unsigned>> WorkList;
  std::vector<SETPass::JumpInfo> &Jumps;
  std::map<GlobalVariable *, uint64_t> CanonicalValues;
};

bool SET::enqueueStores(LoadInst *Start) {
  bool Handled = true;
  unsigned InitialHeight = OS.height();
  auto *Destination = Start->getPointerOperand();
  std::stack<std::pair<Instruction *, unsigned>> ToExplore;
  std::set<BasicBlock *> Visited;
  ToExplore.push(make_pair(Start, 0));

  Instruction *I = Start;

  while (!ToExplore.empty()) {
    unsigned Depth;
    std::tie(I, Depth) = ToExplore.top();
    ToExplore.pop();
    BasicBlock *BB = I->getParent();

    // If we already visited this basic block just skip it and record the fact
    // that we were not able to completely handle the situation.
    if (Visited.count(BB) > 0) {
      Handled = false;
      continue;
    }

    Visited.insert(BB);
    BasicBlock::reverse_iterator It(++I->getReverseIterator());
    BasicBlock::reverse_iterator Begin(BB->rend());

    bool Found = false;
    for (; It != Begin; It++) {
      if (auto *Store = dyn_cast<StoreInst>(&*It)) {
        if (Store->getPointerOperand() == Destination) {
          auto NewPair = make_pair(Store->getValueOperand(), InitialHeight);

          // If a value is already in the work list we might be in a loop, which
          // we don't handle. Note this and don't insert the pair in the work
          // list.
          if (contains(WorkList, NewPair))
            Handled = false;
          else
            WorkList.push_back(NewPair);

          Found = true;
          break;
        }
      }
    }

    // If we haven't find a store, proceed recursively in the predecessors
    if (!Found) {

      // Limit the depth in terms of basic block we're going backwards
      if (Depth >= MaxDepth) {
        Handled = false;
      } else {
        for (BasicBlock *Predecessor : predecessors(BB)) {
          // Skip if the predecessor is the dispatcher
          if (!JTM->isTranslatedBB(Predecessor)) {
            Handled = false;
          } else {
            if (!Predecessor->empty())
              ToExplore.push(make_pair(&*Predecessor->rbegin(), Depth + 1));
          }
        }
      }
    }
  }

  return Handled;
}

void SET::collectMetadata() {
  const Module *M = getModule(&F);
  QuickMetadata QMD(getContext(M));

  // Collect canonical values
  const char *MDName = "revng.input.canonical-values";
  NamedMDNode *CanonicalValuesMD = M->getNamedMetadata(MDName);
  for (MDNode *CanonicalValueMD : CanonicalValuesMD->operands()) {
    auto *CanonicalValueTuple = cast<MDTuple>(CanonicalValueMD);
    auto Name = QMD.extract<StringRef>(CanonicalValueTuple, 0);
    if (GlobalVariable *CSV = M->getGlobalVariable(Name)) {
      uint64_t Value = QMD.extract<uint64_t>(CanonicalValueTuple, 1);
      CanonicalValues[CSV] = Value;
    }
  }
}

bool SET::run() {
  collectMetadata();

  // Run the actual analysis
  for (BasicBlock &BB : make_range(F.begin(), F.end())) {

    if (Visited->find(&BB) != Visited->end())
      continue;
    Visited->insert(&BB);

    for (Instruction &Instr : BB) {
      revng_assert(Instr.getParent() == &BB);

      auto *Store = dyn_cast<StoreInst>(&Instr);
      auto *Load = dyn_cast<LoadInst>(&Instr);
      bool IsStore = Store != nullptr;
      bool IsPCStore = IsStore && JTM->isPCReg(Store->getPointerOperand());

      bool IsLoad = Load != nullptr;
      if ((!IsStore && !IsLoad)
          || (IsPCStore && isa<ConstantInt>(Store->getValueOperand()))
          || (IsLoad
              && (isa<GlobalVariable>(Load->getPointerOperand())
                  || isa<AllocaInst>(Load->getPointerOperand()))))
        continue;

      revng_assert(WorkList.empty());
      if (IsStore) {
        // Clean the OperationsStack and, if we're dealing with a store to the
        // PC, ask it to track all the possible values that the PC will assume.
        OS.reset(Store);
        WorkList.push_back(make_pair(Store->getValueOperand(), 0));
      } else {
        OS.reset();
        WorkList.push_back(make_pair(Load->getPointerOperand(), 0));
      }

      std::set<Value *> Visited;

      while (!WorkList.empty()) {
        unsigned Height;
        Value *V;
        std::tie(V, Height) = WorkList.back();
        WorkList.pop_back();

        if (Visited.find(V) != Visited.end())
          continue;
        Visited.insert(V);

        // Discard operations we no longer need
        OS.cut(Height);

        while (V != nullptr)
          V = handleInstruction(&Instr, V);
      }

      if (IsPCStore && OS.hasTrackedValues()) {
        bool IsApproximate = OS.isApproximate();
        Jumps.emplace_back(Store, IsApproximate, OS.trackedValues());
      }
    }
  }

  OS.registerPCs();
  OS.registerLoadAddresses();

  return false;
}

char SETPass::ID = 0;

void SETPass::getAnalysisUsage(AnalysisUsage &AU) const {
  if (UseOSRA) {
    AU.addRequired<OSRAPass>();
    AU.addRequired<ConditionalReachedLoadsPass>();
  }

  AU.addRequired<FunctionCallIdentification>();
}

bool SETPass::runOnModule(Module &M) {
  revng_log(PassesLog, "Starting SETPass");

  Function &F = *M.getFunction("root");
  freeContainer(Jumps);

  auto *OSRA = getAnalysisIfAvailable<OSRAPass>();
  if (OSRA != nullptr) {
    auto &CRDP = getAnalysis<ConditionalReachedLoadsPass>();
    JTM->noReturn().collectDefinitions(CRDP);
  }

  FunctionCallIdentification &FCI = getAnalysis<FunctionCallIdentification>();

  SET SimpleExpressionTracker(F, JTM, OSRA, &FCI, Visited, Jumps);

  revng_log(PassesLog, "Ending SETPass");
  return SimpleExpressionTracker.run();
}

bool SET::handleInstructionWithOSRA(Instruction *Target, Value *V) {
  revng_assert(OSRA != nullptr);

  // We don't know how to proceed, but we can still check if the current
  // instruction is associated with a suitable OSR
  const OSRAPass::OSR *O = OSRA->getOSR(V);
  using CI = ConstantInt;
  Type *Int64 = IntegerType::get(F.getParent()->getContext(), 64);

  if (O == nullptr || O->boundedValue()->isTop()
      || O->boundedValue()->isBottom()) {
    return false;
  } else if (O->isConstant()) {
    // If it's just a single constant, use it
    OS.explore(CI::get(Int64, O->constant()));
  } else {
    // Hard limit
    if (not O->boundedValue()->hasSignedness() or O->size() >= 10000)
      return false;

    // We have a limited range, let's use it all

    // Perform a preliminary check that whole range fits into the executable
    // area
    Constant *MinConst, *MaxConst;
    std::tie(MinConst, MaxConst) = O->boundaries(Int64, DL);

    // TODO: note that since we check if isExecutableRange, this part will never
    //       affect the noreturn syscalls detection
    // TODO: the O->size() threshold is pretty arbitrary, the best solution
    //       here is probably restore it to int64_t::max(), assert if it's
    //       larger than 10000 and only apply it to store to memory, pc and
    //       maybe other registers (lr?)
    auto MaterializedMin = OS.materializeSimple(MinConst);
    auto MaterializedMax = OS.materializeSimple(MaxConst);
    auto MaterializedStep = OS.materializeSimple(CI::get(Int64, O->factor()));

    if (OS.readsMemory()) {
      // If there's a load in the stack only check the first and last element
      if (!JTM->isPC(MaterializedMin) || !JTM->isPC(MaterializedMax)) {
        return false;
      }
    } else {
      // If there are no loads, the whole range of generated addresses must be
      // executable and properly aligned
      if (!JTM->isExecutableRange(MaterializedMin, MaterializedMax)
          || !JTM->isInstructionAligned(MaterializedStep)) {
        return false;
      }
    }

    if (O->size() > 1000)
      dbg << "Warning: " << O->size() << " jump targets added\n";

    // Note: addition and comparison for equality are all sign-safe
    // operations, no need to use Constants in this case.
    for (uint64_t Address : O->bounds(OS.topType())) {
      OS.explore(CI::get(Int64, Address));
    }
  }

  return true;
}

Value *SET::handleInstruction(Instruction *Target, Value *V) {
  bool Handled = false;

  // Blacklist i128
  // TODO: should we black list all the non-integer types?
  if (V->getType()->isIntegerTy(128))
    return nullptr;

  if (auto *C = dyn_cast<ConstantInt>(V)) {
    // We reached the end of the path, materialize the value
    OS.explore(C);
    return nullptr;
  }

  if (auto *Load = dyn_cast<LoadInst>(V)) {

    //
    // Handle canonical values
    //
    if (auto *Target = dyn_cast<GlobalVariable>(Load->getPointerOperand())) {
      auto It = CanonicalValues.find(Target);
      if (It != CanonicalValues.end()) {
        OS.explore(ConstantInt::get(Load->getType(), It->second));
        // Do not return, proceed as usual
      }
    }
  }

  if (OSRA != nullptr && !OS.empty()) {
    if (handleInstructionWithOSRA(Target, V))
      return nullptr;
  }

  if (auto *BinOp = dyn_cast<BinaryOperator>(V)) {

    // Append a reference to the operation to the Operations stack
    Use &FirstOp = BinOp->getOperandUse(0);
    Use &SecondOp = BinOp->getOperandUse(1);

    bool IsFirstConstant = isa<ConstantInt>(FirstOp.get());
    bool IsSecondConstant = isa<ConstantInt>(SecondOp.get());

    if (IsFirstConstant || IsSecondConstant) {
      revng_assert(!(IsFirstConstant && IsSecondConstant));

      // Add to the operations stack the constant one and proceed with the other
      if (OS.insertIfNew(BinOp))
        return IsFirstConstant ? SecondOp.get() : FirstOp.get();
    } else if (OSRA != nullptr) {
      Constant *ConstantOp = nullptr;
      Value *FreeOp = nullptr;
      std::tie(ConstantOp, FreeOp) = OSRA->identifyOperands(BinOp, DL);

      if (FreeOp == nullptr && ConstantOp != nullptr) {
        // The operation has been folded
        OS.explore(ConstantOp);
        return nullptr;
      } else if (FreeOp != nullptr && ConstantOp != nullptr) {
        // We were able to identify a constant operand
        unsigned FreeOpIndex = BinOp->getOperand(0) == FreeOp ? 0 : 1;

        // Note: the lifetime of the cloned instruction is managed by the
        //       OperationsStack
        Instruction *Clone = BinOp->clone();
        Clone->setOperand(1 - FreeOpIndex, ConstantOp);
        // This is a dirty trick to keep track of the original
        // instruction
        Clone->setOperand(FreeOpIndex, BinOp);

        // TODO: this might leave to infinte loops
        if (OS.insertIfNew(Clone, BinOp))
          return BinOp->getOperandUse(FreeOpIndex).get();
      }
    }
  } else if (auto *Load = dyn_cast<LoadInst>(V)) {
    auto *Pointer = Load->getPointerOperand();

    // If we're loading a global or local variable, look for the last write to
    // that variable, otherwise see if it's a load from a constant address which
    // points to a constant memory area
    if (isa<GlobalVariable>(Pointer) || isa<AllocaInst>(Pointer)) {
      // Enqueue the stores and write down if we were able to handle to writers
      // to this load, if not, we'll let OSRA try, otherwise we'll call
      // OS.setApproximate later
      Handled = enqueueStores(Load);
    } else {
      if (OS.insertIfNew(Load))
        return Pointer;
    }
  } else if (auto *Unary = dyn_cast<UnaryInstruction>(V)) {
    if (OS.insertIfNew(Unary))
      return Unary->getOperand(0);
  } else if (auto *Expression = dyn_cast<ConstantExpr>(V)) {
    if (Expression->getNumOperands() == 1) {
      auto *ExprAsInstr = Expression->getAsInstruction();
      OS.insert(ExprAsInstr);
      return Expression->getOperand(0);
    }
  } else if (auto *Call = dyn_cast<CallInst>(V)) {
    Function *Callee = Call->getCalledFunction();
    if (Callee != nullptr && Callee->getIntrinsicID() == Intrinsic::bswap) {
      OS.insert(Call);
      return Call->getArgOperand(0);
    }
  } else if (auto *Select = dyn_cast<SelectInst>(V)) {
    Value *TrueVal = Select->getTrueValue();
    Value *FalseVal = Select->getFalseValue();
    bool IsTrueConstant = isa<ConstantInt>(TrueVal);
    bool IsFalseConstant = isa<ConstantInt>(FalseVal);

    if (IsTrueConstant or IsFalseConstant) {

      if (IsTrueConstant)
        OS.explore(cast<Constant>(TrueVal));

      if (IsFalseConstant)
        OS.explore(cast<Constant>(FalseVal));

      if (not(IsTrueConstant or IsFalseConstant)) {
        if (OS.insertIfNew(Select))
          return IsTrueConstant ? FalseVal : TrueVal;
      } else {
        return nullptr;
      }
    }
  } // End of the switch over instruction type

  if (!Handled)
    OS.setApproximate();
  return nullptr;
}
