//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"

#include "revng/Support/IRHelpers.h"

#include "revng-c/MakeEnvNull/MakeEnvNull.h"

bool MakeEnvNullPass::runOnModule(llvm::Module &M) {

  llvm::GlobalVariable *Env = M.getGlobalVariable("env",
                                                  /* AllowInternal */ true);

  if (nullptr == Env)
    return false;

  bool Changed = false;
  for (llvm::Function &F : M) {

    if (not F.getMetadata("revng.func.entry"))
      continue;

    auto *EnvType = Env->getType();
    revng_assert(EnvType->isPointerTy());
    auto *Null = llvm::Constant::getNullValue(EnvType);
    if (replaceAllUsesInFunctionWith(&F, Env, Null))
      Changed = true;
  }

  return Changed;
}

char MakeEnvNullPass::ID = 0;

using llvm::RegisterPass;
using Pass = MakeEnvNullPass;
static RegisterPass<Pass> RegisterMakeEnvNull("make-env-null",
                                              "Pass that substitutes env with "
                                              "a null pointer");
