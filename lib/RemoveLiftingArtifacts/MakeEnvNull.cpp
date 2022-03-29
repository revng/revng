//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

struct MakeEnvNullPass : public FunctionPass {
public:
  static char ID;

  MakeEnvNullPass() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoadModelWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    // Skip non-isolated functions
    auto FTags = FunctionTags::TagsSet::from(&F);
    if (not FTags.contains(FunctionTags::Isolated))
      return false;

    bool Changed = false;

    Module *M = F.getParent();
    GlobalVariable *Env = M->getGlobalVariable("env",
                                               /* AllowInternal */ true);

    SmallPtrSet<LoadInst *, 8> LoadsFromEnvInF;
    for (Use &EnvUse : Env->uses()) {

      if (auto *I = dyn_cast<Instruction>(EnvUse.getUser())) {

        if (I->getFunction() != &F)
          continue;

        // At this point, all uses of env in a function should be loads
        LoadsFromEnvInF.insert(cast<LoadInst>(I));

      } else if (auto *CE = dyn_cast<ConstantExpr>(EnvUse.getUser())) {

        if (not CE->isCast())
          continue;

        for (Use &CEUse : CE->uses()) {
          if (auto *I = dyn_cast<Instruction>(CEUse.getUser())) {

            if (I->getFunction() != &F)
              continue;

            // At this point, all uses of env in a function should be loads
            LoadsFromEnvInF.insert(cast<LoadInst>(I));
          }
        }
      }
    }

    for (LoadInst *L : LoadsFromEnvInF) {
      Type *LoadType = L->getType();
      auto *Null = Constant::getNullValue(LoadType);
      L->replaceAllUsesWith(Null);
    }

    Changed = not LoadsFromEnvInF.empty();
    return Changed;
  }
};

char MakeEnvNullPass::ID = 0;

using Reg = RegisterPass<MakeEnvNullPass>;
static Reg RegisterMakeEnvNull("make-env-null",
                               "Pass that substitutes env with a null pointer");
