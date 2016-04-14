/// \file
/// \brief

// LLVM includes
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"

// Local includes
#include "debug.h"
#include "revamb.h"
#include "ir-helpers.h"
#include "osra.h"
#include "jumptargetmanager.h"
#include "set.h"

using namespace llvm;

char SETPass::ID = 0;

void SETPass::getAnalysisUsage(AnalysisUsage &AU) const {
  if (UseOSRA)
    AU.addRequired<OSRAPass>();
}

void SETPass::enqueueStores(LoadInst *Start,
                            unsigned StackHeight,
                            std::vector<std::pair<Value *, unsigned>>& WL) {
  auto *Destination = Start->getPointerOperand();
  std::stack<std::pair<Instruction *, unsigned>> ToExplore;
  std::set<BasicBlock *> Visited;
  ToExplore.push(std::make_pair(Start, 0));

  Instruction *I = Start;

  while (!ToExplore.empty()) {
    unsigned Depth;
    std::tie(I, Depth) = ToExplore.top();
    ToExplore.pop();

    auto *BB = I->getParent();
    if (Visited.find(BB) != Visited.end())
      continue;

    Visited.insert(BB);
    BasicBlock::reverse_iterator It(make_reverse_iterator(I));
    BasicBlock::reverse_iterator Begin(BB->rend());

    bool Found = false;
    for (; It != Begin; It++) {
      if (auto *Store = dyn_cast<StoreInst>(&*It)) {
        if (Store->getPointerOperand() == Destination) {
          auto NewPair = std::make_pair(Store->getValueOperand(), StackHeight);
          if (std::find(WL.begin(), WL.end(), NewPair) == WL.end())
            WL.push_back(NewPair);
          Found = true;
          break;
        }
      }
    }

    // If we haven't find a store, proceed recursively in the predecessors
    if (!Found && Depth < MaxDepth) {
      auto Predecessors = make_range(pred_begin(BB), pred_end(BB));
      for (BasicBlock *Predecessor : Predecessors) {
        if (Predecessor != JTM->dispatcher()
            && !Predecessor->empty()) {
          ToExplore.push(std::make_pair(&*Predecessor->rbegin(),
                                        Depth + 1));
        }
      }
    }

  }

}

class OperationsStack {
public:
  OperationsStack(JumpTargetManager *JTM,
                  const DataLayout &DL) : JTM(JTM), DL(DL) { }

  void explore(Constant *NewOperand);
  uint64_t materialize(Constant *NewOperand);

  void reset(bool Reliable) {
    Operations.clear();
    OperationsSet.clear();
    IsReliable = Reliable;
  }

  void registerPCs() const {
    for (auto Pair : PCs)
      JTM->getBlockAt(Pair.first, Pair.second);
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
      Operations.pop_back();
    }
  }

  bool insertIfNew(Instruction *I) {
    if (OperationsSet.find(I) == OperationsSet.end()) {
      Operations.push_back(I);
      OperationsSet.insert(I);
      return true;
    }
    return false;
  }

  bool insertIfNew(Instruction *I, Instruction *Ref) {
    if (OperationsSet.find(Ref) == OperationsSet.end()) {
      Operations.push_back(I);
      OperationsSet.insert(Ref);
      return true;
    }
    return false;
  }

  void insert(Instruction *I) {
    Operations.push_back(I);
  }

  unsigned height() const { return Operations.size(); }

private:
  JumpTargetManager *JTM;
  const DataLayout &DL;

  std::vector<Instruction *> Operations;
  std::set<Instruction *> OperationsSet;
  std::set<std::pair<uint64_t, bool>> PCs;

  bool IsReliable;
};

uint64_t OperationsStack::materialize(Constant *NewOperand) {
  for (Instruction *I : make_range(Operations.rbegin(), Operations.rend())) {
    if (auto *Load = dyn_cast<LoadInst>(I)) {
      // OK, we've got a load, let's see if the load address is
      // constant
      assert(NewOperand != nullptr && !isa<UndefValue>(NewOperand));

      if (Load->getType()->isIntegerTy()) {
        unsigned Size = Load->getType()->getPrimitiveSizeInBits() / 8;
        assert(Size != 0);
        NewOperand = JTM->readConstantInt(NewOperand, Size);
      } else if (Load->getType()->isPointerTy()) {
        NewOperand = JTM->readConstantPointer(NewOperand,
                                              Load->getType());
      } else {
        assert(false);
      }

      if (NewOperand == nullptr)
        break;
    } else if (auto *Call = dyn_cast<CallInst>(I)) {
      Function *Callee = Call->getCalledFunction();
      assert(Callee != nullptr
             && Callee->getIntrinsicID() == Intrinsic::bswap);
      uint64_t Value = NewOperand->getUniqueInteger().getLimitedValue();

      Type *T = NewOperand->getType();
      if (T->isIntegerTy(16))
        Value = ByteSwap_16(Value);
      else if (T->isIntegerTy(32))
        Value = ByteSwap_32(Value);
      else if (T->isIntegerTy(64))
        Value = ByteSwap_64(Value);

      NewOperand = ConstantInt::get(T, Value);
    } else {
      // Replace non-const operand with NewOperand
      std::vector<Constant *> Operands;
      bool NonConstFound = false;
      for (Value *Op : I->operand_values()) {
        if (auto *Const = dyn_cast<Constant>(Op)) {
          Operands.push_back(Const);
        } else {
          assert(!NonConstFound);
          NonConstFound = true;
          Operands.push_back(NewOperand);
        }
      }

      NewOperand = ConstantFoldInstOperands(I->getOpcode(),
                                            I->getType(),
                                            Operands,
                                            DL);
      assert(NewOperand != nullptr);
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

  if (PC != 0 && JTM->isInterestingPC(PC))
    PCs.insert({ PC, IsReliable });
}

bool SETPass::runOnFunction(Function &F) {
  OSRAPass *OSRA = getAnalysisIfAvailable<OSRAPass>();
  const DataLayout &DL = F.getParent()->getDataLayout();
  OperationsStack OS(JTM, DL);

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

      // Keep this for future use
      bool IsLoad = false && Load != nullptr;
      if ((!IsStore && !IsLoad)
          || (IsPCStore
              && isa<ConstantInt>(Store->getValueOperand()))
          || (IsLoad
              && (isa<GlobalVariable>(Load->getPointerOperand())
                  || isa<AllocaInst>(Load->getPointerOperand()))))
        continue;

      // Operations is a stack of ConstantInt uses in a BinaryOperator
      // TODO: hardcoded
      OS.reset(/* IsPCStore */ false);
      std::vector<std::pair<Value *, unsigned>> WorkList;
      if (IsStore)
        WorkList.push_back(std::make_pair(Store->getValueOperand(), 0));
      else
        WorkList.push_back(std::make_pair(Load->getPointerOperand(), 0));

      std::set<Value *> Visited;

      while (!WorkList.empty()) {
        unsigned Height;
        Value *V;
        std::tie(V, Height) = WorkList.back();
        WorkList.pop_back();
        Value *Next = V;

        if (Visited.find(V) != Visited.end())
          continue;
        Visited.insert(V);

        // Discard operations we no longer need
        OS.cut(Height);

        while (Next != nullptr) {
          V = Next;
          Next = nullptr;

          if (auto *C = dyn_cast<ConstantInt>(V)) {
            // We reached the end of the path, materialize the value
            OS.explore(C);
          } else if (auto *BinOp = dyn_cast<BinaryOperator>(V)) {

            // Append a reference to the operation to the Operations stack
            Use& FirstOp = BinOp->getOperandUse(0);
            Use& SecondOp = BinOp->getOperandUse(1);

            if (isa<ConstantInt>(FirstOp.get())
                || isa<ConstantInt>(SecondOp.get())) {
              assert(!(isa<ConstantInt>(FirstOp.get())
                       && isa<ConstantInt>(SecondOp.get())));
              bool FirstConstant = isa<ConstantInt>(FirstOp.get());

              // Add to the operations stack the constant one and proceed with
              // the other
              if (OS.insertIfNew(BinOp))
                Next = FirstConstant ? SecondOp.get() : FirstOp.get();
            } else if (OSRA != nullptr) {
              Constant *ConstantOp = nullptr;
              Value *FreeOp = nullptr;
              Type *Int64 = Type::getInt64Ty(F.getParent()->getContext());
              std::tie(ConstantOp, FreeOp) = OSRA->identifyOperands(BinOp,
                                                                    Int64,
                                                                    DL);

              if (FreeOp == nullptr && ConstantOp != nullptr) {
                // The operation has been folded
                OS.explore(ConstantOp);
              } else if (FreeOp != nullptr && ConstantOp != nullptr) {
                // We were able to identify a constant operand
                unsigned FreeOpIndex = BinOp->getOperand(0) == FreeOp ? 0 : 1;

                Instruction *Clone = BinOp->clone();
                Clone->setOperand(1 - FreeOpIndex, ConstantOp);
                // This is a dirty trick to keep track of the original
                // instruction
                Clone->setOperand(FreeOpIndex, BinOp);
                // TODO: this might leave to infinte loops
                if (OS.insertIfNew(Clone, BinOp))
                  Next = BinOp->getOperandUse(FreeOpIndex).get();
              }
            }
          } else if (auto *Load = dyn_cast<LoadInst>(V)) {
            auto *Pointer = Load->getPointerOperand();

            // If we're loading a global or local variable, look for the last
            // write to that variable, otherwise see if it's a load from a
            // constant address which points to a constant memory area
            if (isa<GlobalVariable>(Pointer) || isa<AllocaInst>(Pointer)) {
              enqueueStores(Load, OS.height(), WorkList);
            } else {
              if (OS.insertIfNew(Load))
                Next = Pointer;
            }
          } else if (auto *Unary = dyn_cast<UnaryInstruction>(V)) {
            if (OS.insertIfNew(Unary))
              Next = Unary->getOperand(0);
          } else if (auto *Expression = dyn_cast<ConstantExpr>(V)) {
            if (Expression->getNumOperands() == 1) {
              auto *ExprAsInstr = Expression->getAsInstruction();
              OS.insert(ExprAsInstr);
              Next = Expression->getOperand(0);
            }
          } else if (auto *Call = dyn_cast<CallInst>(V)) {
            Function *Callee = Call->getCalledFunction();
            if (Callee != nullptr
                && Callee->getIntrinsicID() == Intrinsic::bswap) {
              OS.insert(Call);
              Next = Call->getArgOperand(0);
            }
          } // End of the switch over instruction type

            // We don't know how to proceed, but we can still check if the
            // current instruction is associated with a suitable OSR.
          if (OSRA != nullptr && Next == nullptr && OS.height() > 0) {
            const OSRAPass::OSR *O = OSRA->getOSR(V);
            if (O == nullptr)
              continue;

            using CI = ConstantInt;
            Type *Int64 = IntegerType::get(F.getParent()->getContext(), 64);
            if (O->isConstant()) {
              // If it's just a single constant, use it
              OS.explore(CI::get(Int64, O->base()));
            } else if (!O->boundedValue()->isTop()
                       && !O->boundedValue()->isBottom()
                       && O->boundedValue()->isSingleRange()) {
              // We have a limited range, let's use it all

              // Perform a preliminary check that whole range fits into the
              // executable area
              Constant *MinConst, *MaxConst;
              std::tie(MinConst, MaxConst) = O->boundaries(Int64, DL);
              uint64_t Min = getZExtValue(MinConst, DL);
              uint64_t Max = getZExtValue(MaxConst, DL);
              // uint64_t Step = O->absFactor(Int64, DL);
              uint64_t Step = O->factor();

              // TODO: the O->size() threshold is pretty arbitrary, the best
              //       solution here is probably restore it to int64_t::max(),
              //       assert if it's larger than 10000 and only apply it to
              //       store to memory, pc and maybe other registers (lr?)
              auto MaterializedMin = OS.materialize(MinConst);
              auto MaterializedMax = OS.materialize(MaxConst);
              auto MaterializedStep = OS.materialize(CI::get(Int64, Step));
              if (!JTM->isExecutableRange(MaterializedMin, MaterializedMax)
                  || !JTM->isInstructionAligned(MaterializedStep)
                  || O->size() >= 10000)
                continue;

              if (O->size() > 1000)
                dbg << "Warning: " << O->size() << " jump targets added\n";

              DBG("osrjts", dbg << "Adding " << std::dec << O->size()
                  << " jump targets from 0x"
                  << std::hex << JTM->getPC(&Instr).first << "\n");

              // Note: addition and comparison for equality are all sign-safe
              // operations, no need to use Constants in this case.
              // TODO: switch to a super-elegant iterator
              for (uint64_t Position = Min; Position != Max; Position += Step)
                OS.explore(CI::get(Int64, Position));
              OS.explore(CI::get(Int64, Max));
            }
          }

        }
      }
    }
  }

  OS.registerPCs();

  return false;
}
