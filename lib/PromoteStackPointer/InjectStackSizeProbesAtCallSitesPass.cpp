//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/IRBuilder.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/PromoteStackPointer/InjectStackSizeProbesAtCallSitesPass.h"

using namespace llvm;

bool InjectStackSizeProbesAtCallSitesPass::runOnModule(llvm::Module &M) {
  bool Changed = false;
  IRBuilder<> B(M.getContext());

  // Get the stack pointer CSV
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
  auto *SP = GCBI.spReg();
  auto *SPType = SP->getValueType();

  // Create marker for recording stack height at each call site
  auto *SSACSType = llvm::FunctionType::get(B.getVoidTy(), { SPType }, false);
  auto SSACS = M.getOrInsertFunction("stack_size_at_call_site", SSACSType);
  auto *F = cast<Function>(SSACS.getCallee());
  F->addFnAttr(Attribute::NoUnwind);
  F->addFnAttr(Attribute::WillReturn);
  F->addFnAttr(Attribute::NoMerge);
  F->setOnlyAccessesInaccessibleMemory();

  for (Function &F : FunctionTags::Isolated.functions(&M)) {
    if (F.isDeclaration())
      continue;
    setInsertPointToFirstNonAlloca(B, F);

    auto *SP0 = createLoad(B, SP);

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (isCallToIsolatedFunction(&I)) {
          // We found a function call
          Changed = true;
          B.SetInsertPoint(&I);

          // Inject a call to the marker. First argument is sp - sp0
          auto *Call = B.CreateCall(SSACS, B.CreateSub(SP0, createLoad(B, SP)));
          Call->copyMetadata(I);
        }
      }
    }
  }

  return Changed;
}

using MSSACSP = InjectStackSizeProbesAtCallSitesPass;
void MSSACSP::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
  AU.setPreservesCFG();
}

char InjectStackSizeProbesAtCallSitesPass::ID = 0;

using RegisterMSSACS = RegisterPass<InjectStackSizeProbesAtCallSitesPass>;
static RegisterMSSACS R("measure-stack-size-at-call-sites",
                        "Measure Stack Size At Call Sites Pass");
