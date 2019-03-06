#include <llvm/ADT/SmallSet.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Module.h>

#include <revng/Support/IRHelpers.h>

#include "DecompilationHelpers.h"

using namespace llvm;

static bool isInAnyFunction(Instruction *I, const std::set<Function *> &Funcs) {
  return Funcs.count(I->getFunction()) != 0;
}

static bool
isUsedInFunction(ConstantExpr *CE, const Function &F) {
  SmallSet<Constant *, 16> UnexploredCEUsers;
  UnexploredCEUsers.insert(CE);
  SmallSet<Constant *, 16> NextUnexploredCEUsers;
  do {
    NextUnexploredCEUsers.clear();
    for (Constant *CEUser : UnexploredCEUsers) {
      for (User *U : CEUser->users()) {

        if (auto *I = dyn_cast<Instruction>(U)) {
          if (I->getFunction() == &F)
            return true;

        } else if (auto *CExpr = dyn_cast<Constant>(U)) {
          NextUnexploredCEUsers.insert(CExpr);

        } else {
          revng_abort();
        }
      }
    }
    std::swap(UnexploredCEUsers, NextUnexploredCEUsers);
  } while (not UnexploredCEUsers.empty());

  return false;
}

std::set<GlobalVariable *> getDirectlyUsedGlobals(Function &F) {
  std::set<GlobalVariable *> Results;
  Module *M =  F.getParent();
  for (GlobalVariable &G : M->globals()) {
    for (User *U : G.users()) {
      if (auto *I = dyn_cast<Instruction>(U)) {
        if (I->getFunction() == &F) {
          Results.insert(&G);
          break;
        }
      } else if (auto *CE = dyn_cast<ConstantExpr>(U)) {
        if (isUsedInFunction(CE, F)) {
          Results.insert(&G);
          break;
        }
      } else {
        revng_abort();
      }
    }
  }
  return Results;
}

std::set<Function *> getDirectlyCalledFunctions(Function &F) {
  std::set<Function *> Results;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *Call = dyn_cast<CallInst>(&I))
        if (Function *Callee = getCallee(Call))
          Results.insert(Callee);
  return Results;
}
