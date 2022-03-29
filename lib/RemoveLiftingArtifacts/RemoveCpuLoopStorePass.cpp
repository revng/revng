//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

class RemoveCpuLoopStorePass : public FunctionPass {
public:
  static char ID;

public:
  RemoveCpuLoopStorePass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {

    // Skip non-isolated functions
    auto FTags = FunctionTags::TagsSet::from(&F);
    if (not FTags.contains(FunctionTags::Isolated))
      return false;

    // Retrieve the global variable `cpu_loop_exiting`
    Module *M = F.getParent();
    GlobalVariable *CpuLoop = M->getGlobalVariable("cpu_loop_exiting");

    // Remove in bulk all the users of the global variable.
    SmallVector<LoadInst *, 8> Loads;
    SmallVector<StoreInst *, 8> Stores;
    for (User *U : CpuLoop->users()) {
      Instruction *I = cast<Instruction>(U);

      // Check only translated functions.
      if (I->getParent()->getParent() != &F)
        continue;

      if (auto *Store = dyn_cast<StoreInst>(U))
        Stores.push_back(Store);
      else if (auto *Load = dyn_cast<LoadInst>(U))
        Loads.push_back(Load);
      else
        revng_abort("Unexpected use of cpu_loop_exiting");
    }

    // Remove in bulk all the store found before.
    bool Changed = not Loads.empty() or not Stores.empty();
    for (Instruction *I : Stores)
      eraseFromParent(I);

    for (LoadInst *L : Loads) {
      // Replace all uses of loads with "false"
      L->replaceAllUsesWith(Constant::getNullValue(L->getType()));
      eraseFromParent(L);
    }

    return Changed;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoadModelWrapperPass>();
  }
};

char RemoveCpuLoopStorePass::ID = 0;
using Reg = RegisterPass<RemoveCpuLoopStorePass>;
static Reg X("remove-cpu-loop-store", "Removes store to cpu_loop", true, true);
