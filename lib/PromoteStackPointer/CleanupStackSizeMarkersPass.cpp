//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Instructions.h"

#include "revng/Model/FunctionTags.h"
#include "revng/PromoteStackPointer/CleanupStackSizeMarkersPass.h"
#include "revng/PromoteStackPointer/InstrumentStackAccessesPass.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

bool CleanupStackSizeMarkersPass::runOnModule(Module &M) {
  SmallVector<CallInst *, 16> CallsToDelete;
  SmallVector<Function *, 16> FunctionsToDelete;

  for (Function &F : FunctionTags::StackOffsetMarker.functions(&M)) {
    for (User *U : F.users()) {
      auto *Call = cast<CallInst>(U);
      Call->replaceAllUsesWith(Call->getArgOperand(0));
      CallsToDelete.push_back(Call);
    }

    FunctionsToDelete.push_back(&F);
  }

  if (auto *SSACS = getIRHelper("stack_size_at_call_site", M)) {
    for (User *U : SSACS->users()) {
      auto *Call = cast<CallInst>(U);
      CallsToDelete.push_back(Call);
    }

    FunctionsToDelete.push_back(SSACS);
  }

  for (CallInst *Call : CallsToDelete) {
    eraseFromParent(Call);
  }

  for (Function *F : FunctionsToDelete) {
    eraseFromParent(F);
  }

  return CallsToDelete.size() + FunctionsToDelete.size() != 0;
}

void CleanupStackSizeMarkersPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
}

char CleanupStackSizeMarkersPass::ID = 0;

using RegisterCSSM = RegisterPass<CleanupStackSizeMarkersPass>;
static RegisterCSSM R("cleanup-stack-size-markers",
                      "Cleanup Stack Size Markers Pass");
