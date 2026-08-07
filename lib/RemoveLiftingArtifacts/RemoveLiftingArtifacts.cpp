//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"

#include "revng/Model/FunctionTags.h"
#include "revng/RemoveLiftingArtifacts/RemoveLiftingArtifacts.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/NewPC.h"

using namespace llvm;

static void removeCallsToArtifacts(Function &F) {
  // Remove calls to `newpc` in the current function.
  SmallVector<Instruction *, 8> ToErase;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *C = dyn_cast<CallInst>(&I))
        if (auto *Callee = getCallee(C)) {
          // Remove calls to newpc and Exceptional functions
          // TODO: we also remove calls to set_PlainMetaAddress since emitting C
          //       structs is currently unsupported by the backend. We should
          //       eventually find a better solution.
          if (NewPCHelper.getCall(C).has_value()
              or Callee->getName() == "set_PlainMetaAddress"
              or FunctionTags::Exceptional.isTagOf(Callee)) {
            ToErase.push_back(C);
          }
        }

      // Remove LLVM debug intrinsics
      if (auto *Dbg = dyn_cast<DbgInfoIntrinsic>(&I))
        ToErase.push_back(Dbg);
    }
  }

  for (Instruction *I : ToErase)
    eraseFromParent(I);
}

static void removeStoresToCPULoopExiting(Function &F) {
  // Retrieve the global variable `cpu_loop_exiting`
  Module *M = F.getParent();
  GlobalVariable *CpuLoop = M->getGlobalVariable("cpu_loop_exiting", true);
  if (CpuLoop == nullptr)
    return;

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
  for (Instruction *I : Stores)
    eraseFromParent(I);

  for (LoadInst *L : Loads) {
    // Replace all uses of loads with "false"
    L->replaceAllUsesWith(Constant::getNullValue(L->getType()));
    eraseFromParent(L);
  }
}

static void makeEnvNull(Function &F) {
  Module *M = F.getParent();
  GlobalVariable *Env = M->getGlobalVariable("env",
                                             /* AllowInternal */ true);

  if (Env == nullptr)
    return;

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
}

static void removeLiftingArtifacts(Function &F) {
  removeCallsToArtifacts(F);
  removeStoresToCPULoopExiting(F);
  makeEnvNull(F);
}

namespace revng::pypeline::piperuns {

void RemoveLiftingArtifacts::runOnLLVMFunction(const model::Function &Function,
                                               llvm::Function &LLVMFunction) {
  llvm::Module &Module = *LLVMFunction.getParent();
  for (llvm::Function &F : Module) {
    if (FunctionTags::Isolated.isTagOf(&F)
        or FunctionTags::SegmentGlobalGetter.isTagOf(&F)) {
      continue;
    }

    // If we find a non-isolated function with body, we want to remove it.
    deleteOnlyBody(F);

    // Mark non-isolated functions as OptimizeNone (optnone).
    // We want all future passes in the decompilation pipeline not to look
    // at non-isolated functions, because it would just be a waste of time,
    // and they might also not respect some of the assumptions the
    // decompilation pipeline makes, causing crashes.
    if (not F.hasFnAttribute(Attribute::AttrKind::OptimizeNone)) {
      F.addFnAttr(Attribute::AttrKind::OptimizeNone);
    }

    // Mark non-isolated functions as NoInline (noinline), since we don't
    // want them to be inlined into isolated functions for some reason.
    if (not F.hasFnAttribute(Attribute::AttrKind::NoInline)) {
      F.addFnAttr(Attribute::AttrKind::NoInline);
    }
  }

  revng_assert(FunctionTags::Isolated.isTagOf(&LLVMFunction));
  removeLiftingArtifacts(LLVMFunction);
  FunctionTags::LiftingArtifactsRemoved.addTo(&LLVMFunction);
}

} // namespace revng::pypeline::piperuns
