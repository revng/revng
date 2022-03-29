//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

class RemoveExceptionCallsPass : public FunctionPass {
public:
  static char ID;

public:
  RemoveExceptionCallsPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    // Skip non-isolated functions
    auto FTags = FunctionTags::TagsSet::from(&F);
    if (not FTags.contains(FunctionTags::Isolated))
      return false;

    // Remove calls to `raise_exception_helper` in the current function.
    SmallVector<Instruction *, 8> ToErase;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB)
        if (auto *C = dyn_cast<CallInst>(&I))
          if (auto *Callee = getCallee(C);
              Callee and Callee->getName() == "raise_exception_helper")
            ToErase.push_back(C);
    }

    bool Changed = not ToErase.empty();
    for (Instruction *I : ToErase)
      eraseFromParent(I);

    return Changed;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoadModelWrapperPass>();
  }
};

char RemoveExceptionCallsPass::ID = 0;
using Reg = RegisterPass<RemoveExceptionCallsPass>;
static Reg X("remove-exception-calls",
             "Removes calls to raise_exception_helper",
             true,
             true);
