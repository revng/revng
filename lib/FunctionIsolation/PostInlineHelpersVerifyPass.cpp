//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include "revng/Model/FunctionTags.h"
#include "revng/Support/Assert.h"
#include "revng/Support/IRHelpers.h"

#include "PostInlineHelpersVerifyPass.h"

using namespace llvm;

bool PostInlineHelpersVerifyPass::runOnModule(Module &M) {
  for (Function &F : M) {
    if (not FunctionTags::Isolated.isTagOf(&F))
      continue;

    for (Instruction &I : instructions(F))
      if (isa<GetElementPtrInst>(&I))
        revng_abort(("post-inline-helpers-verify: "
                     "getelementptr in Isolated function "
                     + F.getName().str())
                      .c_str());
  }

  return false;
}

char PostInlineHelpersVerifyPass::ID;
