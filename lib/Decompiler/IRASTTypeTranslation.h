#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <map>

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"

#include "clang/AST/Type.h"

#include "revng-c/Decompiler/DLALayouts.h"

namespace llvm {
class Function;
class GlobalVariable;
class Type;
class Value;
} // end namespace llvm

namespace clang {
class ASTContext;
class DeclCtx;
class FieldDecl;
class FunctionDecl;
class TypeDecl;
class VarDecl;
} // end namespace clang

namespace IR2AST {
class StmtBuilder;
} // end namespace IR2AST

class DeclCreator {

public:
  using ValueTypeDeclMap = std::map<const llvm::Value *, clang::TypeDecl *>;
  using GlobalsMap = std::map<const llvm::GlobalVariable *, clang::VarDecl *>;
  using FunctionsMap = std::map<llvm::Function *, clang::FunctionDecl *>;
  using TypeDeclMap = std::map<const llvm::Type *, clang::TypeDecl *>;
  using FieldDeclMap = std::map<clang::TypeDecl *,
                                llvm::SmallVector<clang::FieldDecl *, 8>>;

public:
  DeclCreator(const dla::ValueLayoutMap *LM) :
    ValueTypeDecls(),
    TypeDecls(),
    FieldDecls(),
    GlobalDecls(),
    FunctionDecls(),
    ValueLayouts(LM) {}

  clang::QualType getOrCreateBoolQualType(clang::ASTContext &ASTCtx,
                                          const llvm::Type *Ty = nullptr);

  clang::QualType getOrCreateQualType(const llvm::Value *I,
                                      clang::ASTContext &ASTCtx,
                                      clang::DeclContext &DeclCtx);

  clang::QualType getOrCreateQualType(const llvm::GlobalVariable *G,
                                      clang::ASTContext &ASTCtx,
                                      clang::DeclContext &DeclCtx);

  clang::QualType getOrCreateQualType(const llvm::Type *T,
                                      const llvm::Value *NamingValue,
                                      clang::ASTContext &ASTCtx,
                                      clang::DeclContext &DeclCtx);

  void createTypeDeclsForFunctionPrototype(clang::ASTContext &C,
                                           llvm::Function *TheF);

  void createFunctionAndCalleesDecl(clang::ASTContext &C, llvm::Function *TheF);

  void createGlobalVarDeclUsedByFunction(clang::ASTContext &Context,
                                         llvm::Function *TheF,
                                         IR2AST::StmtBuilder &ASTBuilder);

public:
  ValueTypeDeclMap ValueTypeDecls;
  TypeDeclMap TypeDecls;
  FieldDeclMap FieldDecls;
  GlobalsMap GlobalDecls;
  FunctionsMap FunctionDecls;
  const dla::ValueLayoutMap *ValueLayouts;

protected:
  clang::QualType getOrCreateFunctionRetType(llvm::Function *F,
                                             clang::ASTContext &ASTCtx,
                                             clang::DeclContext &DeclCtx) {
    const llvm::FunctionType *FType = F->getFunctionType();
    return getOrCreateQualType(FType->getReturnType(), F, ASTCtx, DeclCtx);
  }

  clang::FunctionDecl *createFunDecl(clang::ASTContext &Context,
                                     llvm::Function *F,
                                     bool IsDefinition);
}; // end class DeclCreator
