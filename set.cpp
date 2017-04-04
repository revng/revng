/// \file set.cpp
/// \brief Simple Expression Tracker pass implementation
/// This file is composed by three main parts: the OperationsStack
/// implementation, the SET algorithm and the SET pass

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <iterator>

// LLVM includes
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"

// Local includes
#include "datastructures.h"
#include "debug.h"
#include "revamb.h"
#include "ir-helpers.h"
#include "osra.h"
#include "jumptargetmanager.h"
#include "set.h"

using namespace llvm;
using std::make_pair;

/// \brief Stack to keep track of the operations generating a specific value
///
/// The OperationsStacks offers the following features:
///
/// * it doesn't insert more than once an item (to avoid loops)
/// * it can traverse the stack from top to bottom to produce a value and, if
///   required register it with the JumpTargetManager
/// * cut the stack to a certain height
/// * manage the lifetime of orphan instruction it contains
/// * keep track of all the possible values assumed since the last reset and
///   whether this information is precise or not
class OperationsStack {
public:
  OperationsStack(JumpTargetManager *JTM,
                  const DataLayout &DL) : JTM(JTM), DL(DL), LoadsCount(0) {
    reset();
  }

  ~OperationsStack() {
    reset();
  }

  void explore(Constant *NewOperand);
  uint64_t materialize(Constant *NewOperand);

  /// \brief What values should be tracked
  enum TrackingType {
    None, ///< Don't track anything
    PCsOnly, ///< Track only values which can be PCs
    All ///< Track all the values
  };

  /// \brief Clean the operations stack
  void reset() {
    // Delete all the temporary instructions we created
    for (Instruction *I : Operations)
      if (I->getParent() == nullptr)
        delete I;

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
    const auto SETToPC = JumpTargetManager::SETToPC;
    const auto SETNotToPC = JumpTargetManager::SETNotToPC;
    for (auto &P : NewPCs)
      JTM->registerJT(P.first, P.second ? SETToPC : SETNotToPC);
  }

  void cut(unsigned Height) {
    assert(Height <= Operations.size());
    while (Height != Operations.size()) {
      Instruction *Op = Operations.back();
      auto It = OperationsSet.find(Op);
      if (It != OperationsSet.end())
        OperationsSet.erase(It);
      else if (isa<BinaryOperator>(Op)) {
        // It's not in OperationsSet, it might a binary instruction where we
        // forced one operand to be constant, or an instruction generated from a
        // constant unary expression
        unsigned FreeOpIndex = isa<Constant>(Op->getOperand(0)) ? 1 : 0;
        auto *FreeOp = cast<Instruction>(Op->getOperand(FreeOpIndex));
        auto It = OperationsSet.find(FreeOp);
        assert(It != OperationsSet.end());
        OperationsSet.erase(It);
      }

      // We have the ownership of instruction without parent
      if (Op->getParent() == nullptr)
        delete Op;

      if (isa<LoadInst>(Op)) {
        assert(LoadsCount > 0);
        LoadsCount--;
      }

      Operations.pop_back();
    }
  }

  bool insertIfNew(Instruction *I) {
    return insertIfNew(I, I);
  }

  bool insertIfNew(Instruction *I, Instruction *Ref) {
    if (OperationsSet.find(Ref) == OperationsSet.end()) {
      insert(I);
      OperationsSet.insert(Ref);
      return true;
    }

    // If the given instruction doesn't have a parent we take ownership of it
    if (I->getParent() == nullptr)
      delete I;

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
  bool empty() const { return height() == 0; }

  Type *topType() const {
    Type *Result = nullptr;
    bool NonConstFound = false;

    for (Value *Op : Operations.back()->operand_values()) {
      if (!isa<Constant>(Op)) {
        assert(!NonConstFound);
        (void) NonConstFound;
        NonConstFound = true;
        Result = Op->getType();
      }
    }

    assert(Result != nullptr);
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

  bool hasTrackedValues() const { return TrackedValues.size() != 0; }

  bool readsMemory() const { return LoadsCount > 0; }

private:
  JumpTargetManager *JTM;
  const DataLayout &DL;

  std::vector<Instruction *> Operations;
  std::set<Instruction *> OperationsSet;
  std::set<std::pair<uint64_t, bool>> NewPCs;
  std::set<uint64_t> TrackedValues;

  bool Approximate;
  TrackingType Tracking;
  bool IsPCStore;
  bool SetsSyscallNumber;
  unsigned LoadsCount;

  Instruction *Target;
};

uint64_t OperationsStack::materialize(Constant *NewOperand) {
  for (Instruction *I : make_range(Operations.rbegin(), Operations.rend())) {
    if (auto *Load = dyn_cast<LoadInst>(I)) {
      // OK, we've got a load, let's see if the load address is
      // constant
      assert(NewOperand != nullptr && !isa<UndefValue>(NewOperand));

      // Read the value using the endianess of the destination architecture,
      // since, if there's a mismatch, in the stack we will also have a byteswap
      // instruction
      JumpTargetManager::Endianess E = JumpTargetManager::DestinationEndianess;
      if (Load->getType()->isIntegerTy()) {
        unsigned Size = Load->getType()->getPrimitiveSizeInBits() / 8;
        assert(Size != 0);
        NewOperand = JTM->readConstantInt(NewOperand, Size, E);
      } else if (Load->getType()->isPointerTy()) {
        NewOperand = JTM->readConstantPointer(NewOperand, Load->getType(), E);
      } else {
        assert(false);
      }

      if (NewOperand == nullptr)
        break;

    } else if (auto *Call = dyn_cast<CallInst>(I)) {
      Function *Callee = Call->getCalledFunction();
      assert(Callee != nullptr && Callee->getIntrinsicID() == Intrinsic::bswap);
      (void) Callee;

      uint64_t Value = NewOperand->getUniqueInteger().getLimitedValue();

      Type *T = NewOperand->getType();
      if (T->isIntegerTy(16))
        Value = ByteSwap_16(Value);
      else if (T->isIntegerTy(32))
        Value = ByteSwap_32(Value);
      else if (T->isIntegerTy(64))
        Value = ByteSwap_64(Value);
      else
        llvm_unreachable("Unexpected type");

      NewOperand = ConstantInt::get(T, Value);
    } else {
      // Replace non-const operand with NewOperand
      std::vector<Constant *> Operands;
      bool NonConstFound = false;
      (void) NonConstFound;

      for (Value *Op : I->operand_values()) {
        if (auto *Const = dyn_cast<Constant>(Op)) {
          Operands.push_back(Const);
        } else {
          assert(!NonConstFound);
          NonConstFound = true;
          NewOperand = ConstantExpr::getTruncOrBitCast(NewOperand,
                                                       Op->getType());
          Operands.push_back(NewOperand);
        }
      }

      NewOperand = ConstantFoldInstOperands(I->getOpcode(),
                                            I->getType(),
                                            Operands,
                                            DL);
      assert(NewOperand != nullptr);
      // TODO: this is an hack hiding a bigger problem
      if (isa<UndefValue>(NewOperand)) {
        NewOperand = nullptr;
        break;
      }
    }
  }

  // We made it, mark the value to be explored
  if (NewOperand != nullptr) {
    assert(!isa<UndefValue>(NewOperand));
    return getZExtValue(NewOperand, DL);
  }

  return 0;
}

void OperationsStack::explore(Constant *NewOperand) {
  uint64_t PC = materialize(NewOperand);

  if (PC != 0 && JTM->isPC(PC))
    NewPCs.insert({ PC, IsPCStore });

  if (PC != 0 && (Tracking == All
                  || (Tracking == PCsOnly && JTM->isPC(PC))))
    TrackedValues.insert(PC);

  if (SetsSyscallNumber) {
    Instruction *Top = Operations.size() == 0 ? Target : Operations.back();

    // TODO: don't ignore temporary instructions
    if (Top->getParent() != nullptr)
      JTM->noReturn().registerKiller(PC, Top, Target);
  }
}

/// \brief Simple Expression Tracker implementation
class SET {

public:
  SET(Function &F,
      JumpTargetManager *JTM,
      OSRAPass *OSRA,
      std::set<BasicBlock *> *Visited,
      std::vector<SETPass::JumpInfo> &Jumps) :
    DL(F.getParent()->getDataLayout()),
    JTM(JTM),
    OS(JTM, DL),
    F(F),
    OSRA(OSRA),
    Visited(Visited),
    Jumps(Jumps) { }

  /// \brief Run the Simple Expression Tracker on F
  bool run();

private:
  /// \brief Enqueue all the store seen by the Start load instruction
  /// \return true if it was possible to fully handle all instruction writing to
  ///         the source of the load instruction.
  bool enqueueStores(LoadInst *Start);

  /// \brief Process \p V
  /// \return a boolean indicating whether V has been handled properly and a new
  ///         Value from which SET should proceed
  Value *handleInstruction(Instruction *Target, Value *V);

  /// \brief Process \p V using information from OSRA
  /// \return true if the instruction was handled.
  bool handleInstructionWithOSRA(Instruction *Target, Value *V);

private:
  const unsigned MaxDepth = 3;
  const DataLayout &DL;
  JumpTargetManager *JTM;
  OperationsStack OS;
  Function& F;
  OSRAPass *OSRA;
  std::set<BasicBlock *> *Visited;
  std::vector<std::pair<Value *, unsigned>> WorkList;
  std::vector<SETPass::JumpInfo> &Jumps;
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
    BasicBlock::reverse_iterator It(make_reverse_iterator(I));
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

bool SET::run() {
  for (BasicBlock& BB : make_range(F.begin(), F.end())) {

    if (Visited->find(&BB) != Visited->end())
      continue;
    Visited->insert(&BB);

    for (Instruction& Instr : BB) {
      assert(Instr.getParent() == &BB);

      auto *Store = dyn_cast<StoreInst>(&Instr);
      auto *Load = dyn_cast<LoadInst>(&Instr);
      bool IsStore = Store != nullptr;
      bool IsPCStore = IsStore && JTM->isPCReg(Store->getPointerOperand());

      // TODO: either drop this or implement blacklisting of loaded locations
      bool IsLoad = false && Load != nullptr;
      if ((!IsStore && !IsLoad)
          || (IsPCStore && isa<ConstantInt>(Store->getValueOperand()))
          || (IsLoad && (isa<GlobalVariable>(Load->getPointerOperand())
                         || isa<AllocaInst>(Load->getPointerOperand()))))
        continue;

      assert(WorkList.empty());
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

      if (IsPCStore && OS.hasTrackedValues())
        Jumps.push_back(SETPass::JumpInfo(Store,
                                          OS.isApproximate(),
                                          OS.trackedValues()));
    }
  }

  OS.registerPCs();

  return false;
}

char SETPass::ID = 0;

void SETPass::getAnalysisUsage(AnalysisUsage &AU) const {
  if (UseOSRA) {
    AU.addRequired<OSRAPass>();
    AU.addRequired<ConditionalReachedLoadsPass>();
  }
}

bool SETPass::runOnFunction(Function &F) {
  DBG("passes", { dbg << "Starting SETPass\n"; });

  freeContainer(Jumps);

  OSRAPass *OSRA = getAnalysisIfAvailable<OSRAPass>();
  if (OSRA != nullptr) {
    auto &CRDP = getAnalysis<ConditionalReachedLoadsPass>();
    JTM->noReturn().collectDefinitions(CRDP);
  }

  SET SimpleExpressionTracker(F, JTM, OSRA, Visited, Jumps);

  DBG("passes", { dbg << "Ending SETPass\n"; });
  return SimpleExpressionTracker.run();
}

bool SET::handleInstructionWithOSRA(Instruction *Target, Value *V) {
  assert(OSRA != nullptr);

  // We don't know how to proceed, but we can still check if the current
  // instruction is associated with a suitable OSR
  const OSRAPass::OSR *O = OSRA->getOSR(V);
  using CI = ConstantInt;
  Type *Int64 = IntegerType::get(F.getParent()->getContext(), 64);

  if (O == nullptr
      || O->boundedValue()->isTop()
      || O->boundedValue()->isBottom()) {
    return false;
  } else if (O->isConstant()) {
    // If it's just a single constant, use it
    OS.explore(CI::get(Int64, O->constant()));
  } else {
    // Hard limit
    if (O->size() >= 10000)
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
    auto MaterializedMin = OS.materialize(MinConst);
    auto MaterializedMax = OS.materialize(MaxConst);
    auto MaterializedStep = OS.materialize(CI::get(Int64, O->factor()));

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

    DBG("osrjts", dbg << "Adding " << std::dec << O->size()
        << " jump targets from 0x"
        << std::hex << JTM->getPC(Target).first << "\n");

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
      assert(!(IsFirstConstant && IsSecondConstant));

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
  } // End of the switch over instruction type

  if (!Handled)
    OS.setApproximate();
  return nullptr;
}
