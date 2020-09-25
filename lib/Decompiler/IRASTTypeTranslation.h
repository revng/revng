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
class Function;
class GlobalVariable;
class Type;
class Value;
class LLVMContext;
} // end namespace llvm

namespace clang {
class ASTContext;
class DeclCtx;
class FunctionDecl;
class TypeDecl;
class VarDecl;
} // end namespace clang

namespace IR2AST {
class StmtBuilder;
} // end namespace IR2AST

class DeclCreator {

public:
  using ValueQualTypeMap = std::map<const llvm::Value *, clang::QualType>;
  using GlobalVarDeclMap = std::map<const llvm::GlobalVariable *,
                                    clang::VarDecl *>;
  using FunctionDeclMap = std::map<const llvm::Function *,
                                   clang::FunctionDecl *>;

  using TypeDeclVec = std::vector<clang::TypeDecl *>;
  using TypeDeclMap = std::map<const llvm::Type *, TypeDeclVec::size_type>;

public:
  DeclCreator(const dla::ValueLayoutMap *LM) :
    ValueLayouts(LM),
    LayoutQualTypes(),
    TypeDecls(),
    TypeDeclsMap(),
    ValueQualTypes(),
    GlobalDecls(),
    FunctionDecls() {}

  clang::QualType getOrCreateBoolQualType(clang::ASTContext &ASTCtx,
                                          const llvm::Type *Ty = nullptr);

  clang::QualType getOrCreateValueQualType(const llvm::Value *V,
                                           clang::ASTContext &ASTCtx,
                                           clang::DeclContext &DeclCtx);

  clang::QualType getOrCreateQualType(const llvm::Type *T,
                                      const llvm::Value *NamingValue,
                                      clang::ASTContext &ASTCtx,
                                      clang::DeclContext &DeclCtx);

  llvm::SmallVector<const dla::Layout *, 16>
  getPointedLayouts(const llvm::Value *V) const;

  llvm::Optional<clang::QualType>
  getOrCreateDLAQualType(const llvm::Value *V,
                         clang::ASTContext &ASTCtx,
                         clang::DeclContext &DeclCtx);

  clang::QualType getOrCreateQualTypeFromLayout(const dla::Layout *L,
                                                clang::ASTContext &ClangCtx,
                                                llvm::LLVMContext &LLVMCtx);

  void createTypeDeclsForFunctionPrototype(clang::ASTContext &C,
                                           const llvm::Function *TheF);

  void createFunctionAndCalleesDecl(clang::ASTContext &C,
                                    const llvm::Function *TheF);

  void createGlobalVarDeclUsedByFunction(clang::ASTContext &Context,
                                         const llvm::Function *TheF,
                                         IR2AST::StmtBuilder &ASTBuilder);

  clang::TypeDecl *getTypeDeclOrNull(const llvm::Type *Ty) const {

    auto It = TypeDeclsMap.find(Ty);

    if (It == TypeDeclsMap.end())
      return nullptr;

    return TypeDecls.at(It->second);
  }

  std::string getUniqueTypeNameForDecl(const llvm::Value *NamingValue) const;

public:
  const auto &typeDeclMap() const { return TypeDeclsMap; }
  const auto &typeDecls() const { return TypeDecls; }
  const auto &globalDecls() const { return GlobalDecls; }
  const auto &functionDecls() const { return FunctionDecls; }

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

  clang::FunctionDecl *createFunDecl(clang::ASTContext &Context,
                                     const llvm::Function *F,
                                     bool IsDefinition);

private:
  const dla::ValueLayoutMap *ValueLayouts;
  std::map<const dla::Layout *, clang::QualType> LayoutQualTypes;
  TypeDeclVec TypeDecls;
  TypeDeclMap TypeDeclsMap;
  ValueQualTypeMap ValueQualTypes;
  GlobalVarDeclMap GlobalDecls;
  FunctionDeclMap FunctionDecls;

}; // end class DeclCreator
