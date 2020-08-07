#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <map>
#include <vector>

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"

#include "clang/AST/Type.h"

#include "revng-c/Decompiler/DLALayouts.h"

namespace llvm {
class Argument;
class Function;
class GlobalVariable;
class Type;
class Value;
class LLVMContext;
} // end namespace llvm

namespace clang {
class ASTContext;
class DeclCtx;
class FieldDecl;
class FunctionDecl;
class TypeDecl;
class Type;
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
  using FunctionRetTypeMap = std::map<llvm::Function *, clang::QualType>;
  using ArgumentTypeMap = std::map<llvm::Argument *, clang::QualType>;
  using FieldDeclMap = std::map<clang::TypeDecl *,
                                llvm::SmallVector<clang::FieldDecl *, 8>>;

public:
  DeclCreator(const dla::ValueLayoutMap *LM) :
    TypeDecls(),
    ValueTypeDecls(),
    FieldDecls(),
    GlobalDecls(),
    FunctionDecls(),
    FunctionRetTypes(),
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

  const auto &typeDeclMap() const { return TypeDeclsMap; }

  const auto &typeDecls() const { return TypeDecls; }

  clang::TypeDecl *getTypeDeclOrNull(const llvm::Type *Ty) const {

    auto It = TypeDeclsMap.find(Ty);

    if (It == TypeDeclsMap.end())
      return nullptr;

    return TypeDecls.at(It->second);
  }

  std::string getUniqueTypeNameForDecl() const {
    return std::string("type_") + std::to_string(TypeDecls.size());
  }

protected:
  bool insertTypeMapping(const llvm::Type *Ty, clang::TypeDecl *TDecl) {

    if (nullptr == Ty or nullptr == TDecl)
      return false;

    auto NewID = TypeDecls.size();
    const auto &[It, NewInsert] = TypeDeclsMap.insert({ Ty, NewID });
    revng_assert(NewInsert or TypeDecls.at(It->second) == TDecl);

    if (NewInsert)
      TypeDecls.push_back(TDecl);

    return NewInsert;
  }

  clang::QualType getOrCreateArgumentType(llvm::Argument *A,
                                          clang::ASTContext &ASTCtx,
                                          clang::DeclContext &DeclCtx);

  clang::QualType getOrCreateFunctionRetType(llvm::Function *F,
                                             clang::ASTContext &ASTCtx,
                                             clang::DeclContext &DeclCtx);

  clang::FunctionDecl *createFunDecl(clang::ASTContext &Context,
                                     llvm::Function *F,
                                     bool IsDefinition);

  clang::QualType getOrCreateQualTypeFromLayout(const dla::Layout *L,
                                                clang::ASTContext &ClangCtx,
                                                llvm::LLVMContext &LLVMCtx);

private:
  using TypeDeclVec = std::vector<clang::TypeDecl *>;
  using TypeDeclMap = std::map<const llvm::Type *, TypeDeclVec::size_type>;
  TypeDeclVec TypeDecls;
  TypeDeclMap TypeDeclsMap;
  std::map<const dla::Layout *, clang::QualType> LayoutQualTypes;

public:
  ValueTypeDeclMap ValueTypeDecls;
  FieldDeclMap FieldDecls;
  GlobalsMap GlobalDecls;
  FunctionsMap FunctionDecls;
  FunctionRetTypeMap FunctionRetTypes;
  ArgumentTypeMap ArgumentTypes;
  const dla::ValueLayoutMap *ValueLayouts;

}; // end class DeclCreator
