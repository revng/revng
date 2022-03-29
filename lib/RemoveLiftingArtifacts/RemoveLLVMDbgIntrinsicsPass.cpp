//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

class RemoveLLVMDbgIntrinsicsPass : public FunctionPass {
public:
  static char ID;

public:
  RemoveLLVMDbgIntrinsicsPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    // Skip non-isolated functions
    auto FTags = FunctionTags::TagsSet::from(&F);
    if (not FTags.contains(FunctionTags::Isolated))
      return false;

    // Remove calls to `llvm.assume` in isolated functions.
    SmallVector<Instruction *, 8> ToErase;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB)
        if (auto *Dbg = dyn_cast<DbgInfoIntrinsic>(&I))
          ToErase.push_back(Dbg);
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

char RemoveLLVMDbgIntrinsicsPass::ID = 0;
using Reg = RegisterPass<RemoveLLVMDbgIntrinsicsPass>;
static Reg
  X("remove-llvm-dbg-intrinsics", "Removes llvm debug intrinsics", true, true);
