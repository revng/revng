/// \file RemoveHelperCalls.cpp
/// Remove calls to helpers in a function and replaces them with stores of an
/// opaque value onto the CSVs clobbered by the helper.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"

#include "revng/BasicAnalyses/RemoveHelperCalls.h"
#include "revng/Support/IRHelpers.h"

class RegisterClobberer {
private:
  llvm::Module *M;
  OpaqueFunctionsPool<std::string> Clobberers;

public:
  RegisterClobberer(llvm::Module *M) : M(M), Clobberers(M, false) {
    using namespace llvm;
    Clobberers.setMemoryEffects(MemoryEffects::readOnly());
    Clobberers.addFnAttribute(Attribute::NoUnwind);
    Clobberers.addFnAttribute(Attribute::WillReturn);
    Clobberers.setTags({ &FunctionTags::ClobbererFunction });
    Clobberers.initializeFromName(FunctionTags::ClobbererFunction);
  }

public:
  llvm::StoreInst *clobber(llvm::IRBuilder<> &Builder,
                           llvm::GlobalVariable *CSV) {
    auto *CSVTy = CSV->getValueType();
    std::string Name = "clobber_" + CSV->getName().str();
    llvm::Function *Clobberer = Clobberers.get(Name, CSVTy, {}, Name);
    return Builder.CreateStore(Builder.CreateCall(Clobberer), CSV);
  }

  llvm::StoreInst *clobber(llvm::IRBuilder<> &Builder,
                           model::Register::Values Value) {
    return clobber(Builder,
                   M->getGlobalVariable(model::Register::getCSVName(Value)));
  }
};

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
      if (isCallToHelper(&I))
        ToReplace.push_back(&I);

  bool Changed = not ToReplace.empty();
  if (!Changed)
    return PreservedAnalyses::all();

  RegisterClobberer Clobberer(F.getParent());
  OpaqueFunctionsPool<Type *> OFPOriginalHelper(F.getParent(), false);
  OFPOriginalHelper.setMemoryEffects(MemoryEffects::readOnly());
  OFPOriginalHelper.addFnAttribute(Attribute::NoUnwind);
  OFPOriginalHelper.addFnAttribute(Attribute::WillReturn);

  IRBuilder<> Builder(F.getContext());
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
