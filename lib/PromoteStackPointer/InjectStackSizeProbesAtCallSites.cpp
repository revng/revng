//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Model/FunctionTags.h"
#include "revng/PromoteStackPointer/InjectStackSizeProbesAtCallSites.h"
#include "revng/Support/IRBuilder.h"

// This name is not present after `CleanupStackSizeMarkers`.
RegisterIRHelper StackSizeAtCallSite("stack_size_at_call_site");

using namespace llvm;

namespace revng::pypeline::piperuns {

void InjectStackSizeProbesAtCallSites::runOnFunction(const model::Function
                                                       &Function) {
  llvm::Module &Module = ModuleContainer.getModule(ObjectID(Function.Entry()));
  GeneratedCodeBasicInfo GCBI(Binary, Module);
  revng::IRBuilder B(Module.getContext());

  // Get the stack pointer CSV
  auto *SP = GCBI.spReg();
  auto *SPType = SP->getValueType();

  // Create marker for recording stack height at each call site
  auto *SSACSType = llvm::FunctionType::get(B.getVoidTy(), { SPType }, false);
  auto SSACS = getOrInsertIRHelper("stack_size_at_call_site",
                                   Module,
                                   SSACSType);
  auto *F = cast<llvm::Function>(SSACS.getCallee());
  F->addFnAttr(Attribute::NoUnwind);
  F->addFnAttr(Attribute::WillReturn);
  F->addFnAttr(Attribute::NoMerge);
  F->setOnlyAccessesInaccessibleMemory();

  for (llvm::Function &F : FunctionTags::Isolated.functions(&Module)) {
    if (F.isDeclaration())
      continue;
    B.setInsertPointToFirstNonAlloca(F);

    auto *SP0 = B.createLoad(SP);

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (isCallToIsolatedFunction(&I)) {
          // We found a function call
          B.SetInsertPoint(&I);

          // Inject a call to the marker. First argument is sp - sp0
          auto *Call = B.CreateCall(SSACS, B.CreateSub(SP0, B.createLoad(SP)));
          Call->copyMetadata(I);
        }
      }
    }
  }
}

} // namespace revng::pypeline::piperuns
