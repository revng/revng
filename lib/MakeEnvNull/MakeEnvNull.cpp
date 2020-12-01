//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include "revng/Support/IRHelpers.h"

#include "revng-c/MakeEnvNull/MakeEnvNull.h"

bool MakeEnvNullPass::runOnFunction(llvm::Function &F) {
  if (not F.getMetadata("revng.func.entry"))
    return false;

  llvm::Module *M = F.getParent();
  llvm::GlobalVariable *Env = M->getGlobalVariable("env",
                                                   /* AllowInternal */ true);

  auto *EnvType = Env->getType();
  revng_assert(EnvType->isPointerTy());
  auto *Null = llvm::Constant::getNullValue(EnvType);
  if (replaceAllUsesInFunctionWith(&F, Env, Null))
    return true;

  return false;
}

char MakeEnvNullPass::ID = 0;

using llvm::RegisterPass;
using Pass = MakeEnvNullPass;
static RegisterPass<Pass> RegisterMakeEnvNull("make-env-null",
                                              "Pass that substitutes env with "
                                              "a null pointer");
