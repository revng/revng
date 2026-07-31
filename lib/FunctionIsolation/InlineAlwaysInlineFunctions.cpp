//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "revng/FunctionIsolation/InlineAlwaysInlineFunctions.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

namespace revng::pypeline::piperuns {

void InlineAlwaysInlineFunctions::runOnLLVMFunction(const model::Function
                                                      &Function,
                                                    llvm::Function
                                                      &LLVMFunction) {
  Module &TheModule = *LLVMFunction.getParent();

  // The only isolated functions with a body in this module, other than the
  // module's own function, are the ones isolate emitted for us to inline
  SmallVector<llvm::Function *> ToInline;
  for (llvm::Function &F : TheModule.functions()) {
    if (&F == &LLVMFunction or F.isDeclaration())
      continue;

    if (FunctionTags::Isolated.isTagOf(&F))
      ToInline.push_back(&F);
  }

  // A function to inline can call another one, hence the fixed point
  bool Changed = true;
  while (Changed) {
    Changed = false;

    for (llvm::Function *F : ToInline) {
      SmallVector<CallBase *> Callers;
      for (CallBase *Caller : callers(F))
        Callers.push_back(Caller);

      for (CallBase *Caller : Callers) {
        InlineFunctionInfo IFI;
        bool Success = InlineFunction(*Caller, IFI).isSuccess();
        revng_assert(Success);
        Changed = true;
      }
    }
  }

  for (llvm::Function *F : ToInline) {
    revng_assert(F->use_empty());
    eraseFromParent(F);
  }
}

} // namespace revng::pypeline::piperuns
