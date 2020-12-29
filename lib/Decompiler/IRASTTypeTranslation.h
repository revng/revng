#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <map>
#include <variant>
#include <vector>

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/GlobalVariable.h"

#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"

#include "revng-c/Decompiler/DLALayouts.h"

namespace llvm {
class Type;
class Value;
class LLVMContext;
} // end namespace llvm

namespace clang {
class ASTContext;
class DeclCtx;
class DeclaratorDecl;
class FunctionDecl;
class TypeDecl;
} // end namespace clang

namespace IR2AST {
class StmtBuilder;
} // end namespace IR2AST

class DeclCreator {

public:
  using ValueQualTypeMap = std::map<const llvm::Value *, clang::QualType>;
  using GlobalDeclMap = std::map<const llvm::GlobalObject *,
                                 clang::DeclaratorDecl *>;

  using Typeable = std::variant<const llvm::Type *, const llvm::Value *>;
  using TypeDeclVec = std::vector<clang::TypeDecl *>;
  using TypeDeclMap = std::map<Typeable, TypeDeclVec::size_type>;

public:
  DeclCreator(const dla::ValueLayoutMap *LM) :
    ValueLayouts(LM),
    LayoutQualTypes(),
    TypeDecls(),
    TypeDeclsMap(),
    ValueQualTypes(),
    GlobalDecls() {}

public:
  const auto &typeDeclMap() const { return TypeDeclsMap; }
  const auto &typeDecls() const { return TypeDecls; }
  const auto &globalDecls() const { return GlobalDecls; }

public:
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

  clang::FunctionDecl &getFunctionDecl(const llvm::Function *F) {
    return *cast<clang::FunctionDecl>(globalDecls().at(F));
  }

  clang::VarDecl &getGlobalVarDecl(const llvm::GlobalVariable *G) {
    return *cast<clang::VarDecl>(globalDecls().at(G));
  }

  clang::TypeDecl *getTypeDeclOrNull(const Typeable &V) const {

    auto It = TypeDeclsMap.find(V);
    if (It == TypeDeclsMap.end())
      return nullptr;

    return TypeDecls.at(It->second);
  }

  std::string getUniqueTypeNameForDecl(const llvm::Value *NamingValue) const;

protected:
  bool insertTypeMapping(const Typeable &V, clang::TypeDecl *TDecl) {

    const auto GetPtr = [](const auto *Ptr) -> const void * { return Ptr; };

    if (nullptr == std::visit(GetPtr, V) or nullptr == TDecl)
      return false;

    auto NewID = TypeDecls.size();
    const auto &[It, NewInsert] = TypeDeclsMap.insert({ V, NewID });
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
  GlobalDeclMap GlobalDecls;

}; // end class DeclCreator
