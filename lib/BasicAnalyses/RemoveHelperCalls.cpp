/// \file RemoveHelperCalls.cpp
/// Remove calls to helpers in a function and replaces them with stores of an
/// opaque value onto the CSVs clobbered by the helper.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"

#include "revng/BasicAnalyses/RemoveHelperCalls.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/OpaqueRegisterUser.h"

static bool isCallToAbort(const llvm::Instruction *I) {
  const llvm::Function *Callee = getCallee(I);
  return Callee and (Callee->getName() == AbortFunctionName);
}

llvm::PreservedAnalyses
RemoveHelperCallsPass::run(llvm::Function &F,
                           llvm::FunctionAnalysisManager &FAM) {
  using namespace llvm;

  // Get the result of the GCBI analysis
  GCBI = &(FAM.getResult<GeneratedCodeBasicInfoAnalysis>(F));
  revng_assert(GCBI != nullptr);

  SmallVector<Instruction *, 16> ToReplace;
  for (auto &BB : F)
    for (auto &I : BB)
      if (isCallToHelper(&I) and not isCallToAbort(&I))
        ToReplace.push_back(&I);

  bool Changed = not ToReplace.empty();
  if (!Changed)
    return PreservedAnalyses::all();

  OpaqueRegisterUser Clobberer(F.getParent());
  OpaqueFunctionsPool<Type *> OFPOriginalHelper(F.getParent(), false);
  OFPOriginalHelper.setMemoryEffects(MemoryEffects::readOnly());
  OFPOriginalHelper.addFnAttribute(Attribute::NoUnwind);
  OFPOriginalHelper.addFnAttribute(Attribute::WillReturn);
  OFPOriginalHelper.setTags({ &FunctionTags::UniquedByPrototype });

  // TODO: the checks should be enabled conditionally based on the user.
  revng::NonDebugInfoCheckingIRBuilder Builder(F.getContext());
  for (auto *I : ToReplace) {
    Builder.SetInsertPoint(I);

    // Assumption: helpers do not leave the stack altered, thus we can save the
    // stack pointer and restore it back later.
    auto *SP = createLoad(Builder, GCBI->spReg());

    auto *RetTy = cast<CallInst>(I)->getFunctionType()->getReturnType();
    auto *OriginalHelperMarker = OFPOriginalHelper.get(RetTy,
                                                       RetTy,
                                                       {},
                                                       "original_helper");

    // Create opaque helper for the original helper and taint the registers
    // originally clobbered.
    CallInst *NewHelper = Builder.CreateCall(OriginalHelperMarker);

    for (auto *CSV : getCSVUsedByHelperCall(I).Written)
      Clobberer.clobber(Builder, CSV);

    // Restore stack pointer back.
    Builder.CreateStore(SP, GCBI->spReg());

    I->replaceAllUsesWith(NewHelper);
    I->eraseFromParent();
  }

  return PreservedAnalyses::none();
}
