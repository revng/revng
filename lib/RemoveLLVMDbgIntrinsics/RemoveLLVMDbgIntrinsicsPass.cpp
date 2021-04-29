//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/RemoveLLVMDbgIntrinsics/RemoveLLVMDbgIntrinsicsPass.h"

using namespace llvm;

using RemoveDbgPass = RemoveLLVMDbgIntrinsicsPass;

char RemoveDbgPass::ID = 0;
using Reg = RegisterPass<RemoveDbgPass>;
static Reg
  X("remove-llvm-dbg-intrinsics", "Removes llvm debug intrinsics", true, true);

void RemoveDbgPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<LoadModelWrapperPass>();
}

bool RemoveDbgPass::runOnFunction(Function &F) {

  // Skip non-isolated functions
  auto FTags = FunctionTags::TagsSet::from(&F);
  if (not FTags.contains(FunctionTags::Lifted))
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
    I->eraseFromParent();

  return Changed;
}
