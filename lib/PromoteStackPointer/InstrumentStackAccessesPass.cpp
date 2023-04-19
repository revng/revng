//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"

#include "revng/ADT/Queue.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Support/OpaqueFunctionsPool.h"

#include "revng-c/PromoteStackPointer/InstrumentStackAccessesPass.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/IRHelpers.h"

using namespace llvm;

static Logger<> Log("instrument-stack-accesses");

class InstrumentStackAccesses {
private:
  OpaqueFunctionsPool<Type *> StackOffsetPool;
  CallInst *SP0 = nullptr;
  Type *SPType = nullptr;

public:
  InstrumentStackAccesses(Module &M) : StackOffsetPool(&M, false) {
    StackOffsetPool.addFnAttribute(Attribute::NoUnwind);
    StackOffsetPool.addFnAttribute(Attribute::WillReturn);
    StackOffsetPool.setMemoryEffects(MemoryEffects::inaccessibleMemOnly());
    StackOffsetPool.setTags({ &FunctionTags::StackOffsetMarker });
  }

public:
  void run(Function &F) {
    revng_log(Log, "Instrumenting " << F.getName());
    LoggerIndent<> Indent(Log);

    reset();

    std::set<Instruction *> StackMemoryAccesses = findStackAccesses(F);

    if (SP0 == nullptr)
      return;

    for (Instruction *I : StackMemoryAccesses)
      instrumentStackAccess(I);
  }

private:
  void reset() {
    SP0 = nullptr;
    SPType = nullptr;
  }

  std::set<Instruction *> findStackAccesses(Function &F);
  void instrumentStackAccess(Instruction *I);
};

std::set<Instruction *>
InstrumentStackAccesses::findStackAccesses(Function &F) {
  // Identify memory accesses whose pointer operand depends on SP
  std::set<Instruction *> StackMemoryAccesses;

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {

      // Is this the call to revng_init_local_sp?
      if (auto *Call = getCallTo(&I, "revng_init_local_sp")) {
        revng_log(Log, "Found call to revng_init_local_sp: " << getName(Call));

        SP0 = Call;
        SPType = SP0->getType();
        // We found a load from SP: look for memory accesses using its value
        // to compute the pointer operand
        OnceQueue<Instruction *> Queue;

        auto EnqueueUsers = [&Queue](Instruction *I) {
          for (User *U : I->users()) {
            if (auto *Store = dyn_cast<StoreInst>(U)) {
              // Enqueue store only if the user is the pointer operand
              if (Store->getPointerOperand() == I) {
                Queue.insert(Store);
              }
            } else if (auto *UI = dyn_cast<Instruction>(U)) {
              Queue.insert(UI);
            }
          }
        };

        EnqueueUsers(Call);

        while (not Queue.empty()) {
          Instruction *I = Queue.pop();

          if (isa<LoadInst>(I) or isa<StoreInst>(I)) {
            revng_log(Log, "Record stack access: " << getName(I));
            StackMemoryAccesses.insert(I);
          }

          // Enqueue users (unless it's load)
          if (not isa<LoadInst>(I))
            EnqueueUsers(I);
        }
      }
    }
  }

  revng_log(Log, StackMemoryAccesses.size() << " memory accesses recorded");

  return StackMemoryAccesses;
}

void InstrumentStackAccesses::instrumentStackAccess(Instruction *I) {
  auto *Pointer = cast<Instruction>(getPointer(I));

  auto *IntegerPointer = cast<Instruction>(skipCasts(Pointer));
  IRBuilder<> B(I);

  if (not IntegerPointer->getType()->isIntegerTy()) {
    Value *PtrToInt = B.CreatePtrToInt(IntegerPointer, SPType);
    IntegerPointer = cast<Instruction>(PtrToInt);
  }

  // Compute offset from SP0
  // Note: here we're using `(x + - SP0)` instead of `x - SP0` since otherwise
  //       certain optimization passes cannot correctly reassociate the sum and
  //       elide SP0.
  Value *StackOffset = B.CreateAdd(B.CreateZExtOrTrunc(IntegerPointer, SPType),
                                   B.CreateNeg(SP0));

  // Get/create an identity function taking as second argument the stack
  // offset
  Type *PointerTy = Pointer->getType();
  auto *StackOffsetFunction = StackOffsetPool.get(PointerTy,
                                                  PointerTy,
                                                  { PointerTy, SPType, SPType },
                                                  "stack_offset");

  // Emit the call
  auto *AccessSize = ConstantInt::get(SPType, getMemoryAccessSize(I));
  auto *ID = B.CreateCall(StackOffsetFunction,
                          { Pointer,
                            StackOffset,
                            B.CreateAdd(StackOffset, AccessSize) });

  // Set the pointer operand
  if (auto *Load = dyn_cast<LoadInst>(I)) {
    Load->setOperand(LoadInst::getPointerOperandIndex(), ID);
  } else if (auto *Store = dyn_cast<StoreInst>(I)) {
    Store->setOperand(StoreInst::getPointerOperandIndex(), ID);
  } else {
    revng_abort();
  }
}

bool InstrumentStackAccessesPass::runOnModule(Module &M) {
  InstrumentStackAccesses Instrumenter(M);

  for (Function &F : FunctionTags::Isolated.functions(&M))
    Instrumenter.run(F);

  return true;
}

void InstrumentStackAccessesPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
}

char InstrumentStackAccessesPass::ID = 0;

using RegisterISA = RegisterPass<InstrumentStackAccessesPass>;
static RegisterISA
  R("instrument-stack-accesses", "Instrument Stack Accesses Pass");
