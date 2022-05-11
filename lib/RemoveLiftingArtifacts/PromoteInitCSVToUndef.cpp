//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/IRHelpers.h"

using namespace llvm;

static bool makeInitRegsUndef(Function &F) {
  bool Changed = false;

  for (auto &BB : F) {
    for (auto &I : BB) {
      auto *Call = dyn_cast<CallInst>(&I);
      if (not Call)
        continue;

      auto *Callee = Call->getCalledFunction();
      if (not Callee or not FunctionTags::OpaqueCSVValue.isTagOf(Callee))
        continue;

      Call->replaceAllUsesWith(llvm::UndefValue::get(Call->getType()));

      Changed = true;
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
