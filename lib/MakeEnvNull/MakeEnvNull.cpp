//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/IsolatedFunctions/IsolatedFunctions.h"
#include "revng-c/MakeEnvNull/MakeEnvNull.h"

void MakeEnvNullPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<LoadModelWrapperPass>();
}

bool MakeEnvNullPass::runOnFunction(llvm::Function &F) {

  // Skip non-isolated functions
  const model::Binary
    &Model = getAnalysis<LoadModelWrapperPass>().get().getReadOnlyModel();
  if (not hasIsolatedFunction(Model, F))
    return false;

  bool Changed = false;

  llvm::Module *M = F.getParent();
  llvm::GlobalVariable *Env = M->getGlobalVariable("env",
                                                   /* AllowInternal */ true);

  llvm::SmallPtrSet<llvm::LoadInst *, 8> LoadsFromEnvInF;
  for (llvm::Use &EnvUse : Env->uses()) {

    if (auto *I = llvm::dyn_cast<llvm::Instruction>(EnvUse.getUser())) {

      if (I->getFunction() != &F)
        continue;

      // At this point, all uses of env in a function should be loads
      LoadsFromEnvInF.insert(llvm::cast<llvm::LoadInst>(I));

    } else if (auto *CE = dyn_cast<llvm::ConstantExpr>(EnvUse.getUser())) {

      if (not CE->isCast())
        continue;

      for (llvm::Use &CEUse : CE->uses()) {
        if (auto *I = llvm::dyn_cast<llvm::Instruction>(CEUse.getUser())) {

          if (I->getFunction() != &F)
            continue;

          // At this point, all uses of env in a function should be loads
          LoadsFromEnvInF.insert(llvm::cast<llvm::LoadInst>(I));
        }
      }
    }
  }

  for (llvm::LoadInst *L : LoadsFromEnvInF) {
    llvm::Type *LoadType = L->getType();
    auto *Null = llvm::Constant::getNullValue(LoadType);
    L->replaceAllUsesWith(Null);
  }

  Changed = not LoadsFromEnvInF.empty();
  return Changed;
}

char MakeEnvNullPass::ID = 0;

using llvm::RegisterPass;
using Pass = MakeEnvNullPass;
static RegisterPass<Pass> RegisterMakeEnvNull("make-env-null",
                                              "Pass that substitutes env with "
                                              "a null pointer");
