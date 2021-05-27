//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"

#include "revng/Support/IRHelpers.h"

#include "DecompilationHelpers.h"

using namespace llvm;

static bool isUsedInFunction(const ConstantExpr *CE, const Function &F) {
  SmallSet<const Constant *, 16> UnexploredCEUsers;
  UnexploredCEUsers.insert(CE);
  SmallSet<const Constant *, 16> NextUnexploredCEUsers;
  do {
    NextUnexploredCEUsers.clear();
    for (const Constant *CEUser : UnexploredCEUsers) {
      for (const User *U : CEUser->users()) {

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

std::set<const GlobalVariable *> getDirectlyUsedGlobals(const Function &F) {
  std::set<const GlobalVariable *> Results;
  const llvm::Module *M = F.getParent();
  for (const GlobalVariable &G : M->globals()) {
    for (const User *U : G.users()) {
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

std::set<const Function *> getDirectlyCalledFunctions(Function &F) {
  std::set<const Function *> Results;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *Call = dyn_cast<CallInst>(&I)) {
        if (auto *Callee = getCallee(Call))
          Results.insert(Callee);
      } else if (isa<UnreachableInst>(&I)) {
        // UnreachableInst are decompiled as calls to abort, so if F has an
        // unreachable instruction we need to add "abort" to the called
        // functions.
        LLVMContext &Ctx = F.getContext();
        Type *Void = Type::getVoidTy(Ctx);
        auto Abort = F.getParent()->getOrInsertFunction("abort", Void);
        Results.insert(cast<Function>(Abort.getCallee()));
      }
    }
  }
  return Results;
}

clang::CastExpr *createCast(clang::QualType LHSQualTy,
                            clang::Expr *RHS,
                            clang::ASTContext &ASTCtx) {
  using namespace clang;
  QualType RHSQualTy = RHS->getType();
  const clang::Type *LHSTy = LHSQualTy.getTypePtr();
  const clang::Type *RHSTy = RHSQualTy.getTypePtr();

  if (isa<clang::ConditionalOperator>(RHS))
    RHS = new (ASTCtx) ParenExpr({}, {}, RHS);

  CastKind CK;
  if (LHSTy->isBooleanType() and RHSTy->isIntegerType()) {
    // casting integer to boolean
    return ImplicitCastExpr::Create(ASTCtx,
                                    LHSQualTy,
                                    CastKind::CK_IntegralToBoolean,
                                    RHS,
                                    nullptr,
                                    VK_RValue,
                                    FPOptions());
  }
  if (RHSTy->isBooleanType() and LHSTy->isIntegerType()) {
    // casting boolean to inteeger
    return ImplicitCastExpr::Create(ASTCtx,
                                    LHSQualTy,
                                    CastKind::CK_IntegralCast,
                                    RHS,
                                    nullptr,
                                    VK_RValue,
                                    FPOptions());
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
      revng_assert((PtrSize >= IntegerSize) or (IntegerSize == 128));
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
                                FPOptions(),
                                TI,
                                {},
                                {});
}
