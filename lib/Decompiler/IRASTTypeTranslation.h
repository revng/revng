#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <map>

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"

#include "clang/AST/Type.h"

namespace llvm {
class Function;
class GlobalVariable;
class Type;
class Value;
} // end namespace llvm

namespace clang {
class FieldDecl;
class FunctionDecl;
class TypeDecl;
class VarDecl;
} // end namespace clang

class IRASTTypeTranslator {

public:
  using ValueTypeDeclMap = std::map<const llvm::Value *, clang::TypeDecl *>;
  using GlobalsMap = std::map<const llvm::GlobalVariable *, clang::VarDecl *>;
  using FunctionsMap = std::map<llvm::Function *, clang::FunctionDecl *>;
  using TypeDeclMap = std::map<const llvm::Type *, clang::TypeDecl *>;
  using FieldDeclMap = std::map<clang::TypeDecl *,
                                llvm::SmallVector<clang::FieldDecl *, 8>>;

public:
  IRASTTypeTranslator() = default;

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

public:
  ValueTypeDeclMap ValueTypeDecls;
  TypeDeclMap TypeDecls;
  FieldDeclMap FieldDecls;
  GlobalsMap GlobalDecls;
  FunctionsMap FunctionDecls;
}; // end class IRASTTypeTranslation
