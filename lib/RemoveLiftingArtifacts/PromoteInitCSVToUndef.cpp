//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include "revng/Model/FunctionTags.h"
#include "revng/Model/NamedTypedRegister.h"
#include "revng/Model/Register.h"
#include "revng/RemoveLiftingArtifacts/PromoteInitCSVToUndef.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

static bool
undefPreservedRegistersInitialization(Function &F,
                                      const model::Function &ModelFunction,
                                      const model::Binary &Binary) {
  bool Changed = false;
  QuickMetadata QMD(F.getParent()->getContext());

  for (auto &BB : F) {
    auto It = BB.begin();
    auto End = BB.end();
    while (It != End) {
      auto Next = std::next(It);

      if (auto *Call = dyn_cast<CallInst>(&*It)) {
        auto *Callee = getCalledFunction(Call);

        const char *MDName = "revng.abi_register";

        if (Callee and FunctionTags::OpaqueCSVValue.isTagOf(Callee)
            and Callee->hasMetadata(MDName)) {
          using namespace model;
          auto *Tuple = cast<MDTuple>(Callee->getMetadata(MDName));
          auto RegisterName = QMD.extract<StringRef>(Tuple, 0);
          auto Register = Register::fromName(RegisterName);
          revng_check(Register != Register::Invalid);

          auto Architecture = Register::getReferenceArchitecture(Register);

          using namespace Architecture;
          if (Register != getReturnAddressRegister(Architecture)) {
            Call->replaceAllUsesWith(llvm::UndefValue::get(Call->getType()));
            Call->eraseFromParent();
            Changed = true;
          }
        }
      }

      It = Next;
    }
  }

  return Changed;
}

namespace revng::pypeline::piperuns {

void PromoteInitCSVToUndef::runOnLLVMFunction(const model::Function &Function,
                                              llvm::Function &LLVMFunction) {
  undefPreservedRegistersInitialization(LLVMFunction, Function, Binary);
}

} // namespace revng::pypeline::piperuns
