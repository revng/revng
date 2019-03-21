#ifndef REVNGC_DECOMPILATION_HELPERS_H
#define REVNGC_DECOMPILATION_HELPERS_H

// std includes
#include <set>

// clang includes
#include <clang/AST/Type.h>

namespace llvm {
class Function;
class GlobalVariable;
} // end namespace llvm

namespace clang {
class ASTContext;
class CStyleCastExpr;
class Expr;
} // end namespace clang

std::set<llvm::GlobalVariable *> getDirectlyUsedGlobals(llvm::Function &F);

std::set<llvm::Function *> getDirectlyCalledFunctions(llvm::Function &F);

clang::CStyleCastExpr *createCast(clang::QualType LHSQualTy,
                                  clang::Expr *RHS,
                                  clang::ASTContext &ASTCtx);

#endif // REVNGC_DECOMPILATION_HELPERS_H
