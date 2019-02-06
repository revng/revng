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
isUsedInFunctions(ConstantExpr *CE, const std::set<Function *> &Funcs) {
  SmallSet<Constant *, 16> UnexploredCEUsers;
  UnexploredCEUsers.insert(CE);
  SmallSet<Constant *, 16> NextUnexploredCEUsers;
  do {
    NextUnexploredCEUsers.clear();
    for (Constant *CEUser : UnexploredCEUsers) {
      for (User *U : CEUser->users()) {

        if (auto *I = dyn_cast<Instruction>(U)) {
          if (isInAnyFunction(I, Funcs))
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

std::set<GlobalVariable *>
getDirectlyUsedGlobals(const std::set<Function *> &Funcs) {
  if (Funcs.empty())
    return {};

  Module *M = (*Funcs.begin())->getParent();
  revng_assert(std::all_of(Funcs.begin(), Funcs.end(), [M](Function *F) {
    return F->getParent() == M;
  }));

  std::set<GlobalVariable *> Results;
  for (GlobalVariable &G : M->globals()) {
    for (User *U : G.users()) {
      if (auto *I = dyn_cast<Instruction>(U)) {
        if (isInAnyFunction(I, Funcs)) {
          Results.insert(&G);
          break;
        }
      } else if (auto *CE = dyn_cast<ConstantExpr>(U)) {
        if (isUsedInFunctions(CE, Funcs)) {
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

std::set<Function *>
getDirectlyCalledFunctions(const std::set<Function *> &Funcs) {
  std::set<Function *> Results;
  for (Function *F : Funcs)
    for (BasicBlock &BB : *F)
      for (Instruction &I : BB)
        if (auto *Call = dyn_cast<CallInst>(&I))
          if (Function *Callee = getCallee(Call))
            Results.insert(Callee);
  return Results;
}

std::set<Function *>
getRecursivelyCalledFunctions(const std::set<Function *> &Funcs) {
  std::set<Function *> Results = Funcs;
  bool NewInsertions;
  do {
    NewInsertions = false;
    for (Function *F : getDirectlyCalledFunctions(Results))
      NewInsertions |= Results.insert(F).second;

  } while (NewInsertions);
  return Results;
}

std::set<Function *> getIsolatedFunctions(llvm::Module &M) {
  std::set<Function *> Result;
  for (Function &F : M.functions()) {
    const llvm::StringRef FName = F.getName();
    bool IsIsolatedFunction = FName.startswith("bb.");
    if (IsIsolatedFunction)
      Result.insert(&F);
  }
  return Result;
}
