#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
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

#include "revng-c/DataLayoutAnalysis/DLALayouts.h"

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

  using TypeDeclOrQualType = std::variant<clang::TypeDecl *, clang::QualType>;
  using TypeVec = std::vector<TypeDeclOrQualType>;

  static clang::QualType getQualType(const TypeDeclOrQualType &T) {

    struct QualTypeGetter {

      clang::QualType operator()(clang::TypeDecl *TD) const {
        if (auto *Typedef = llvm::dyn_cast<clang::TypedefNameDecl>(TD))
          return Typedef->getASTContext().getTypedefType(Typedef);
        return clang::QualType(TD->getTypeForDecl(), 0);
      }

      clang::QualType operator()(clang::QualType QT) const { return QT; }
    };

    return std::visit(QualTypeGetter{}, T);
  }

  static clang::TypeDecl *getTypeDecl(const TypeDeclOrQualType &T) {

    struct TypeDeclGetter {

      clang::TypeDecl *operator()(clang::TypeDecl *TD) const { return TD; }

      clang::TypeDecl *operator()(clang::QualType QT) const { return nullptr; }
    };

    return std::visit(TypeDeclGetter{}, T);
  }

private:
  template<typename... Args>
  using variant = std::variant<Args...>;

public:
  using Typeable = variant<const llvm::Type *,
                           const llvm::Value *,
                           const dla::Layout *>;
  using TypeDeclMap = std::map<Typeable, TypeVec::size_type>;

public:
  DeclCreator(const dla::ValueLayoutMap *LM) :
    ValueLayouts(LM), Types(), TypeDeclsMap(), GlobalDecls() {}

public:
  const auto &types() const { return Types; }
  const auto &globalDecls() const { return GlobalDecls; }

public:
  TypeDeclOrQualType getOrCreateBoolType(clang::ASTContext &ASTCtx,
                                         const llvm::Type *Ty = nullptr);

  TypeDeclOrQualType getOrCreateType(const llvm::Value *V,
                                     clang::ASTContext &ASTCtx,
                                     clang::DeclContext &DeclCtx);

  TypeDeclOrQualType getOrCreateType(const llvm::Type *T,
                                     const llvm::Value *NamingValue,
                                     clang::ASTContext &ASTCtx,
                                     clang::DeclContext &DeclCtx,
                                     bool AllowArbitraryBitSize = false);

  llvm::SmallVector<const dla::Layout *, 16>
  getPointedLayouts(const llvm::Value *V) const;

  llvm::Optional<TypeDeclOrQualType>
  getOrCreateDLAType(const llvm::Value *V,
                     clang::ASTContext &ASTCtx,
                     clang::DeclContext &DeclCtx);

  TypeDeclOrQualType getOrCreateTypeFromLayout(const dla::Layout *L,
                                               clang::ASTContext &ClangCtx,
                                               llvm::LLVMContext &LLVMCtx);

  clang::FunctionDecl &getFunctionDecl(const llvm::Function *F) {
    return *cast<clang::FunctionDecl>(globalDecls().at(F));
  }

  clang::TypeDecl *lookupTypeDeclOrNull(const Typeable &V) const {

    auto It = TypeDeclsMap.find(V);
    if (It == TypeDeclsMap.end())
      return nullptr;

    return getTypeDecl(Types.at(It->second));
  }

  llvm::Optional<TypeDeclOrQualType> lookupType(const Typeable &V) const {

    auto It = TypeDeclsMap.find(V);
    if (It == TypeDeclsMap.end())
      return llvm::None;

    return Types.at(It->second);
  }

  // ---- Functions to call from outside to build global declarations ----

  void createTypeDeclsForFunctionPrototype(clang::ASTContext &C,
                                           const llvm::Function *TheF);

  void createFunctionAndCalleesDecl(clang::ASTContext &C, llvm::Function *TheF);

  void createGlobalVarDeclUsedByFunction(clang::ASTContext &Context,
                                         const llvm::Function *TheF,
                                         IR2AST::StmtBuilder &ASTBuilder);

protected:
  std::string getUniqueTypeNameForDecl(const llvm::Value *NamingValue) const;

protected:
  template<bool NullableTy = false>
  bool insertTypeMapping(const Typeable &V, const TypeDeclOrQualType TDecl) {

    if constexpr (not NullableTy) {
      const auto GetPtr = [](const auto *Ptr) -> const void * { return Ptr; };
      revng_assert(nullptr != std::visit(GetPtr, V));
    }

    auto NewID = Types.size();
    const auto &[It, NewInsert] = TypeDeclsMap.insert({ V, NewID });
    revng_assert(NewInsert or Types.at(It->second) == TDecl);

    if (NewInsert)
      Types.push_back(TDecl);

    return NewInsert;
  }

  TypeDeclOrQualType createTypeFromLayout(const dla::Layout *L,
                                          clang::ASTContext &ClangCtx,
                                          llvm::LLVMContext &LLVMCtx);

  clang::FunctionDecl *createFunDecl(clang::ASTContext &Context,
                                     const llvm::Function *F,
                                     bool IsDefinition);

private:
  const dla::ValueLayoutMap *ValueLayouts;

  TypeVec Types;
  TypeDeclMap TypeDeclsMap;
  GlobalDeclMap GlobalDecls;

}; // end class DeclCreator
