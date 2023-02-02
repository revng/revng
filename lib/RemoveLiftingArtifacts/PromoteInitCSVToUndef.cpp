//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/IRHelpers.h"

using namespace llvm;

static bool makeInitRegsUndef(Function &F) {
  bool Changed = false;

  for (auto &BB : F) {
    auto It = BB.begin();
    auto End = BB.end();
    while (It != End) {
      auto Next = std::next(It);

      if (auto *Call = dyn_cast<CallInst>(&*It)) {
        auto *Callee = Call->getCalledFunction();
        if (Callee and FunctionTags::OpaqueCSVValue.isTagOf(Callee)) {
          Call->replaceAllUsesWith(llvm::UndefValue::get(Call->getType()));
          Call->eraseFromParent();
          Changed = true;
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

  bool runOnFunction(Function &F) override {
    if (FunctionTags::Isolated.isTagOf(&F))
      return makeInitRegsUndef(F);

    return false;
  }
};

char PromoteInitCSVToUndefPass::ID = 0;

static constexpr const char *Flag = "promote-init-csv-to-undef";

using Reg = RegisterPass<PromoteInitCSVToUndefPass>;
static Reg X(Flag, "Promotes calls to init_* functions for CSV to undefs");
