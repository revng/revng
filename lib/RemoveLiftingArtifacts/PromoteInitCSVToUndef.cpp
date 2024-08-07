//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Model/NamedTypedRegister.h"
#include "revng/Model/Register.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/IRHelpers.h"

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

struct PromoteInitCSVToUndefPass : public FunctionPass {
public:
  static char ID;

  PromoteInitCSVToUndefPass() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoadModelWrapperPass>();
    AU.setPreservesCFG();
  }

  bool runOnFunction(Function &F) override {
    if (FunctionTags::Isolated.isTagOf(&F)) {
      auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();
      const model::Binary &Binary = *ModelWrapper.getReadOnlyModel();
      MetaAddress Entry = getMetaAddressMetadata(&F, "revng.function.entry");
      auto &ModelFunction = Binary.Functions().at(Entry);
      return undefPreservedRegistersInitialization(F, ModelFunction, Binary);
    }

    return false;
  }
};

char PromoteInitCSVToUndefPass::ID = 0;

static constexpr const char *Flag = "promote-init-csv-to-undef";

using Reg = RegisterPass<PromoteInitCSVToUndefPass>;
static Reg X(Flag, "Promotes calls to init_* functions for CSV to undefs");
