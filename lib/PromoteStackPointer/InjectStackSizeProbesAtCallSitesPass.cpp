//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Model/FunctionTags.h"
#include "revng/PromoteStackPointer/InjectStackSizeProbesAtCallSites.h"
#include "revng/PromoteStackPointer/InjectStackSizeProbesAtCallSitesPass.h"
#include "revng/Support/IRBuilder.h"

// This name is not present after `CleanupStackSizeMarkers`.
RegisterIRHelper StackSizeAtCallSite("stack_size_at_call_site");

using namespace llvm;

static bool injectStackSizeProbesAtCallSites(llvm::Module &M,
                                             GeneratedCodeBasicInfo &GCBI) {
  bool Changed = false;
  revng::IRBuilder B(M.getContext());

  // Get the stack pointer CSV
  auto *SP = GCBI.spReg();
  auto *SPType = SP->getValueType();

  // Create marker for recording stack height at each call site
  auto *SSACSType = llvm::FunctionType::get(B.getVoidTy(), { SPType }, false);
  auto SSACS = getOrInsertIRHelper("stack_size_at_call_site", M, SSACSType);
  auto *F = cast<Function>(SSACS.getCallee());
  F->addFnAttr(Attribute::NoUnwind);
  F->addFnAttr(Attribute::WillReturn);
  F->addFnAttr(Attribute::NoMerge);
  F->setOnlyAccessesInaccessibleMemory();

  for (Function &F : FunctionTags::Isolated.functions(&M)) {
    if (F.isDeclaration())
      continue;
    B.setInsertPointToFirstNonAlloca(F);

    auto *SP0 = B.createLoad(SP);

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (isCallToIsolatedFunction(&I)) {
          // We found a function call
          Changed = true;
          B.SetInsertPoint(&I);

          // Inject a call to the marker. First argument is sp - sp0
          auto *Call = B.CreateCall(SSACS, B.CreateSub(SP0, B.createLoad(SP)));
          Call->copyMetadata(I);
        }
      }
    }
  }

  return Changed;
}

bool InjectStackSizeProbesAtCallSitesPass::runOnModule(llvm::Module &M) {
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
  return injectStackSizeProbesAtCallSites(M, GCBI);
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

namespace revng::pypeline::piperuns {

// TODO: inline injectStackSizeProbesAtCallSites once we dismiss the old
//       pipeline
void InjectStackSizeProbesAtCallSites::runOnFunction(const model::Function
                                                       &Function) {
  llvm::Module &Module = ModuleContainer.getModule(ObjectID(Function.Entry()));
  GeneratedCodeBasicInfo GCBI(Binary, Module);
  injectStackSizeProbesAtCallSites(Module, GCBI);
}

} // namespace revng::pypeline::piperuns
