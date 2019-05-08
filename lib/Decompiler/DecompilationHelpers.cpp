// LLVM includes
#include <llvm/ADT/SmallSet.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Module.h>

// clang includes
#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>

// revng includes
#include <revng/Support/IRHelpers.h>

// local includes
#include "DecompilationHelpers.h"

using namespace llvm;
using namespace clang;

static bool isInAnyFunction(Instruction *I, const std::set<Function *> &Funcs) {
  return Funcs.count(I->getFunction()) != 0;
}

static bool isUsedInFunction(ConstantExpr *CE, const Function &F) {
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
  llvm::Module *M = F.getParent();
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

clang::CastExpr *createCast(QualType LHSQualTy, Expr *RHS, ASTContext &ASTCtx) {
  QualType RHSQualTy = RHS->getType();
  const clang::Type *LHSTy = LHSQualTy.getTypePtr();
  const clang::Type *RHSTy = RHSQualTy.getTypePtr();

  CastKind CK;
  if (LHSTy->isBooleanType() and RHSTy->isIntegerType()) {
    // casting integer to boolean
    return ImplicitCastExpr::Create(ASTCtx,
                                    LHSQualTy,
                                    CastKind::CK_IntegralToBoolean,
                                    RHS,
                                    nullptr,
                                    VK_RValue);
  }
  if (RHSTy->isBooleanType() and LHSTy->isIntegerType()) {
    // casting boolean to inteeger
    return ImplicitCastExpr::Create(ASTCtx,
                                    LHSQualTy,
                                    CastKind::CK_IntegralCast,
                                    RHS,
                                    nullptr,
                                    VK_RValue);
  }
  if (LHSTy->isIntegerType()) {
    if (RHSTy->isIntegerType()) {
      CK = CastKind::CK_IntegralCast;
    } else if (RHSTy->isPointerType()) {
      CK = CastKind::CK_PointerToIntegral;
    } else {
      revng_abort();
    }
  } else if (LHSTy->isPointerType()) {
    if (RHSTy->isIntegerType()) {

      uint64_t PtrSize = ASTCtx.getTypeSize(LHSQualTy);
      uint64_t IntegerSize = ASTCtx.getTypeSize(RHSQualTy);
      revng_assert(PtrSize >= IntegerSize);
      if (PtrSize > IntegerSize)
        RHS = createCast(ASTCtx.getUIntPtrType(), RHS, ASTCtx);

      CK = CastKind::CK_IntegralToPointer;
    } else if (RHSTy->isPointerType()) {
      CK = CastKind::CK_BitCast;
    } else {
      revng_abort();
    }
  } else {
    revng_abort();
  }
  TypeSourceInfo *TI = ASTCtx.CreateTypeSourceInfo(LHSQualTy);
  return CStyleCastExpr::Create(ASTCtx,
                                LHSQualTy,
                                VK_RValue,
                                CK,
                                RHS,
                                nullptr,
                                TI,
                                {},
                                {});
}
