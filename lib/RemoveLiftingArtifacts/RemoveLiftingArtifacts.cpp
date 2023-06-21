//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Pipeline/RegisterLLVMPass.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/Pipes/Kinds.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/IRHelpers.h"

using namespace llvm;

static bool removeCallsToArtifacts(Function &F) {
  // Remove calls to `newpc` in the current function.
  SmallVector<Instruction *, 8> ToErase;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *C = dyn_cast<CallInst>(&I))
        if (auto *Callee = getCallee(C)) {
          // remove calls to newpc
          if (Callee->getName() == "newpc")
            ToErase.push_back(C);

          // remove exception calls
          if (Callee->getName() == "raise_exception_helper")
            ToErase.push_back(C);
        }

      // remove llvm debug intrisics
      if (auto *Dbg = dyn_cast<DbgInfoIntrinsic>(&I))
        ToErase.push_back(Dbg);
    }
  }

  bool Changed = not ToErase.empty();
  for (Instruction *I : ToErase)
    eraseFromParent(I);

  return Changed;
}

static bool removeStoresToCPULoopExiting(Function &F) {
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

  bool Changed = not Loads.empty() or not Stores.empty();

  // Remove in bulk all the store found before.
  for (Instruction *I : Stores)
    eraseFromParent(I);

  for (LoadInst *L : Loads) {
    // Replace all uses of loads with "false"
    L->replaceAllUsesWith(Constant::getNullValue(L->getType()));
    eraseFromParent(L);
  }

  return Changed;
}

static bool makeEnvNull(Function &F) {

  Module *M = F.getParent();
  GlobalVariable *Env = M->getGlobalVariable("env",
                                             /* AllowInternal */ true);

  if (Env == nullptr)
    return false;

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

  bool Changed = not LoadsFromEnvInF.empty();

  for (LoadInst *L : LoadsFromEnvInF) {
    Type *LoadType = L->getType();
    auto *Null = Constant::getNullValue(LoadType);
    L->replaceAllUsesWith(Null);
  }

  return Changed;
}

static bool removeLiftingArtifacts(Function &F) {
  bool Changed = removeCallsToArtifacts(F);
  Changed |= removeStoresToCPULoopExiting(F);
  Changed |= makeEnvNull(F);
  return Changed;
}

struct RemoveLiftingArtifacts : public ModulePass {
public:
  static char ID;

  RemoveLiftingArtifacts() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    bool Changed = false;
    for (Function &F : M) {
      if (FunctionTags::Isolated.isTagOf(&F)) {
        Changed |= removeLiftingArtifacts(F);
        FunctionTags::LiftingArtifactsRemoved.addTo(&F);
      } else {

        // If we find a non-isolated function with body, we want to remove it.
        Changed |= deleteOnlyBody(F);

        // Mark non-isolated functions as OptimizeNone (optnone).
        // We want all future passes in the decompilation pipeline not to look
        // at non-isolated functions, because it would just be a waste of time,
        // and they might also not respect some of the assumptions the
        // decompilation pipeline makes, causing crashes.
        if (not F.hasFnAttribute(Attribute::AttrKind::OptimizeNone)) {
          F.addFnAttr(Attribute::AttrKind::OptimizeNone);
          Changed = true;
        }

        // Mark non-isolated functions as NoInline (noinline), since we don't
        // want them to be inlined into isolated functions for some reason.
        if (not F.hasFnAttribute(Attribute::AttrKind::NoInline)) {
          F.addFnAttr(Attribute::AttrKind::NoInline);
          Changed = true;
        }
      }
    }

    return Changed;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoadModelWrapperPass>();
  }
};

char RemoveLiftingArtifacts::ID = 0;

static constexpr const char *Flag = "remove-lifting-artifacts";

using Reg = RegisterPass<RemoveLiftingArtifacts>;
static Reg X(Flag,
             "Delete the body of all non-isolated functions, and remove all "
             "lifting artifacts from isolated functions");

struct RemoveLiftingArtifactsPipe {
  static constexpr auto Name = Flag;

  std::vector<pipeline::ContractGroup> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;
    const auto &Removed = LiftingArtifactsRemoved;
    return { ContractGroup::transformOnlyArgument(CSVsPromoted,
                                                  Removed,
                                                  InputPreservation::Erase) };
  }

  void registerPasses(legacy::PassManager &Manager) {
    Manager.add(new RemoveLiftingArtifacts());
  }
};

static pipeline::RegisterLLVMPass<RemoveLiftingArtifactsPipe> Y;
