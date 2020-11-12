#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <set>

#include "clang/AST/Type.h"

namespace llvm {
class Function;
class GlobalVariable;
} // end namespace llvm

namespace clang {
class ASTContext;
class CastExpr;
class Expr;
} // end namespace clang

std::set<llvm::GlobalVariable *> getDirectlyUsedGlobals(llvm::Function &F);

std::set<llvm::Function *> getDirectlyCalledFunctions(llvm::Function &F);

clang::CastExpr *createCast(clang::QualType LHSQualTy,
                            clang::Expr *RHS,
                            clang::ASTContext &ASTCtx);
