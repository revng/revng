//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/Passes/PassBuilder.h"

#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"

using namespace llvm;

static bool isLastBeforeTerminator(Instruction *I) {
  auto It = I->getIterator();
  ++It;
  auto End = I->getParent()->end();
  return It != End && ++It == End;
}

class HoistStructPhis : public llvm::FunctionPass {
public:
  static char ID;
  HoistStructPhis() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override {
    if (not FunctionTags::Isolated.isTagOf(&F))
      return false;

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

    // Create all the phis
    CallInst *FirstCall = nullptr;
    Value *CalledValue = nullptr;
    llvm::SmallVector<Value *, 2> Phis;
    llvm::SmallVector<CallInst *, 2> Calls;
    for (auto [V, Predecessor] : zip(Phi->incoming_values(), Phi->blocks())) {
      // We only expect pure calls
      auto *Call = cast<CallInst>(V);
      Calls.push_back(Call);

      if (CalledValue == nullptr) {
        // This is the first call we see, make some extra checks
        CalledValue = Call->getCalledOperand();
        FirstCall = Call;

        Function *Callee = Call->getCalledFunction();
        revng_assert(Callee != nullptr);

        revng_assert(isLastBeforeTerminator(Call) or Callee->onlyReadsMemory());

        // Ignore isolated functions
        if (FunctionTags::Isolated.isTagOf(Callee))
          return;

        // First iteration, create phis
        for (Type *ArgumentType : Call->getFunctionType()->params())
          Phis.push_back(PHINode::Create(ArgumentType, PhiSize, "", Phi));

      } else {
        // Ensure all the incomings are calls to the same function
        revng_assert(CalledValue == Call->getCalledOperand());
      }

      // Add an incoming
      revng_assert(Call->arg_size() == Phis.size());
      for (auto [Argument, Phi] : zip(Call->args(), Phis))
        cast<PHINode>(Phi)->addIncoming(Argument, Predecessor);
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
      Call->eraseFromParent();
  }
};

char HoistStructPhis::ID;
static RegisterPass<HoistStructPhis> R("hoist-struct-phis", "", false, false);
