#ifndef REVNGC_IRASTTYPETRANSLATION_H
#define REVNGC_IRASTTYPETRANSLATION_H

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <map>

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"

#include "clang/AST/Type.h"

namespace llvm {
class GlobalVariable;
class Type;
class Value;
} // end namespace llvm

namespace clang {
class FieldDecl;
class TypeDecl;
} // end namespace clang

namespace IRASTTypeTranslation {

using TypeDeclMap = std::map<const llvm::Type *, clang::TypeDecl *>;
using FieldDeclMap = std::map<clang::TypeDecl *,
                              llvm::SmallVector<clang::FieldDecl *, 8>>;

clang::QualType getOrCreateBoolQualType(clang::ASTContext &ASTCtx,
                                        TypeDeclMap &TypeDecls,
                                        const llvm::Type *Ty = nullptr);

clang::QualType getOrCreateQualType(const llvm::Value *I,
                                    clang::ASTContext &ASTCtx,
                                    clang::DeclContext &DeclCtx,
                                    TypeDeclMap &TypeDecls,
                                    FieldDeclMap &FieldDecls);

clang::QualType getOrCreateQualType(const llvm::GlobalVariable *G,
                                    clang::ASTContext &ASTCtx,
                                    clang::DeclContext &DeclCtx,
                                    TypeDeclMap &TypeDecls,
                                    FieldDeclMap &FieldDecls);

clang::QualType getOrCreateQualType(const llvm::Type *T,
                                    const llvm::Value *NamingValue,
                                    clang::ASTContext &ASTCtx,
                                    clang::DeclContext &DeclCtx,
                                    TypeDeclMap &TypeDecls,
                                    FieldDeclMap &FieldDecls);

} // end namespace IRASTTypeTranslation

#endif // REVNGC_IRASTTYPETRANSLATION_H
