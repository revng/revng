//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Passes/PassBuilder.h"

#include "revng/Model/FunctionTags.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

static bool isLastBeforeTerminator(Instruction *I) {
  auto It = I->getIterator();
  ++It;
  auto End = I->getParent()->end();
  return It != End && ++It == End;
}

/// This pass hoists calls turns phis of calls returning a `StructType` into
/// a single call (in place of the original phi) whose arguments are phis of the
/// arguments of the original call.
/// This is currently necessary since the backend does not handle well
// `StructType`.
///
/// It turns:
///
/// a:
///   %x1 = call x(1, 2)
///   br c
/// b:
///   %x2 = call x(3, 4)
///   br c
/// c:
///   %x3 = phi [(a, %x1), (b, %x2)]
///
/// Into:
/// c:
///   %arg1 = phi [(a, 1), (b, 3)]
///   %arg2 = phi [(a, 2), (b, 4)]
///   %x3 = call x(%arg1, %arg2)
///
/// \note This can be done only when x does not have side-effects.
class HoistStructPhis : public llvm::FunctionPass {
public:
  static char ID;
  HoistStructPhis() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override {
    llvm::SmallVector<PHINode *, 16> ToFix;

    // Collect phis that need fixing
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *Phi = dyn_cast<PHINode>(&I))
          if (isa<StructType>(I.getType()))
            ToFix.push_back(Phi);

    if (ToFix.size() == 0)
      return false;

    for (PHINode *Phi : ToFix)
      handlePhi(Phi);

    return true;
  }

  void handlePhi(PHINode *Phi) {
    auto PhiSize = Phi->getNumIncomingValues();

    if (PhiSize == 1) {
      Phi->replaceAllUsesWith(Phi->getIncomingValue(0));
      return;
    }

    // Create all the phis
    CallInst *FirstCall = nullptr;
    Value *CalledValue = nullptr;
    llvm::SmallVector<Value *, 2> Phis;
    llvm::SmallVector<CallInst *, 2> Calls;

    for (auto &V : Phi->incoming_values()) {
      if (auto *Call = dyn_cast<CallInst>(V.get())) {
        if (CalledValue == nullptr) {
          // This is the first call we see, make some extra checks
          CalledValue = Call->getCalledOperand();
          FirstCall = Call;

          Function *Callee = getCalledFunction(Call);

          // Ignore isolated functions, they are not side-effect free
          if (Callee == nullptr or FunctionTags::Isolated.isTagOf(Callee))
            return;

          revng_assert(isLastBeforeTerminator(Call)
                       or Callee->onlyReadsMemory());

          // First iteration, create phis
          for (Type *ArgumentType : Call->getFunctionType()->params()) {
            llvm::Instruction *NewPhi = PHINode::Create(ArgumentType,
                                                        PhiSize,
                                                        "",
                                                        Phi);
            NewPhi->setDebugLoc(Phi->getDebugLoc());
            Phis.push_back(NewPhi);
          }

          Calls.push_back(Call);

        } else {
          // Ensure all the incomings are calls to the same function
          revng_assert(CalledValue == Call->getCalledOperand());
        }
      }
    }

    for (auto &&[V, Predecessor] : zip(Phi->incoming_values(), Phi->blocks())) {
      if (isa<UndefValue>(V)) {
        for (auto *NewPhi : Phis)
          cast<PHINode>(NewPhi)->addIncoming(UndefValue::get(NewPhi->getType()),
                                             Predecessor);
      } else if (isa<PoisonValue>(V)) {
        for (auto *NewPhi : Phis) {
          auto *Poison = PoisonValue::get(NewPhi->getType());
          cast<PHINode>(NewPhi)->addIncoming(Poison, Predecessor);
        }
      } else if (auto *Call = dyn_cast<CallInst>(V)) {
        revng_assert(Call->arg_size() == Phis.size());
        for (auto &&[Argument, NewPhi] : zip(Call->args(), Phis))
          cast<PHINode>(NewPhi)->addIncoming(Argument, Predecessor);
      }
    }

    // Create a new function call
    Instruction *InsertionPoint = Phi->getParent()->getFirstNonPHI();
    auto *NewCall = CallInst::Create({ FirstCall->getFunctionType(),
                                       CalledValue },
                                     ArrayRef<Value *>(Phis),
                                     ArrayRef<OperandBundleDef>{},
                                     "",
                                     InsertionPoint);

    // Steal metadata and replace original phi
    NewCall->copyMetadata(*FirstCall);
    revng_assert(NewCall->getType() == Phi->getType());
    Phi->replaceAllUsesWith(NewCall);
    Phi->eraseFromParent();

    for (CallInst *Call : Calls)
      eraseFromParent(Call);
  }
};

char HoistStructPhis::ID;
static RegisterPass<HoistStructPhis> R("hoist-struct-phis", "", false, false);
