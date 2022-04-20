//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <bit>

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"

#include "revng/Support/Assert.h"

#include "revng-c/DataLayoutAnalysis/DLALayouts.h"
#include "revng-c/Support/Mangling.h"

#include "DecompilationHelpers.h"
#include "IRASTTypeTranslation.h"

namespace clang {
class TranslationUnitDecl;
} // end namespace clang

using clang::QualType;

// TODO: this should be a RecursiveCoroutine
DeclCreator::TypeDeclOrQualType
DeclCreator::createTypeFromLayout(const dla::Layout *L,
                                  clang::ASTContext &ClangCtx,
                                  llvm::LLVMContext &LLVMCtx) {
  TypeDeclOrQualType Result = ClangCtx.VoidTy;

  using LayoutKind = dla::Layout::LayoutKind;
  revng_assert(L->size());

  // Build the type decl for the C type associated to layout.
  switch (L->getKind()) {

  case LayoutKind::Padding: {
    QualType CharTy = ClangCtx.CharTy;

    if (L->size() > 1) {

      auto ByteSize = L->size();
      llvm::APInt ArraySize(64 /* bits */, ByteSize);
      auto ArraySizeKind = clang::ArrayType::ArraySizeModifier::Normal;

      Result = ClangCtx.getConstantArrayType(CharTy,
                                             ArraySize,
                                             nullptr,
                                             ArraySizeKind,
                                             0);

    } else {
      Result = CharTy;
    }

  } break;

  case LayoutKind::Base: {
    auto *Base = llvm::cast<dla::BaseLayout>(L);
    auto ByteSize = Base->size();
    revng_assert(ByteSize);
    revng_assert(std::has_single_bit(ByteSize));
    revng_assert(ByteSize <= 16);

    if (Base->PointeeLayout) {
      // If `PointeeLayout` is not null, this is a pointer
      TypeDeclOrQualType EQTy = getOrCreateTypeFromLayout(Base->PointeeLayout,
                                                          ClangCtx,
                                                          LLVMCtx);
      QualType ElemQualTy = DeclCreator::getQualType(EQTy);
      Result = ClangCtx.getPointerType(ElemQualTy);
    } else {
      auto *IntTy = llvm::IntegerType::get(LLVMCtx, ByteSize * 8);
      Result = getOrCreateType(IntTy,
                               nullptr,
                               ClangCtx,
                               *ClangCtx.getTranslationUnitDecl());
    }
  } break;

  case LayoutKind::Array: {

    auto *Array = llvm::cast<dla::ArrayLayout>(L);
    dla::Layout *ElementLayout = Array->getElem();

    TypeDeclOrQualType EQTy = getOrCreateTypeFromLayout(ElementLayout,
                                                        ClangCtx,
                                                        LLVMCtx);
    QualType ElemQualTy = DeclCreator::getQualType(EQTy);

    auto ArraySizeKind = clang::ArrayType::ArraySizeModifier::Normal;
    if (Array->hasKnownLength()) {
      llvm::APInt ArraySize(64 /* bits */, Array->length());
      Result = ClangCtx.getConstantArrayType(ElemQualTy,
                                             ArraySize,
                                             nullptr,
                                             ArraySizeKind,
                                             0);
    } else {
      Result = ClangCtx.getIncompleteArrayType(ElemQualTy, ArraySizeKind, 0);
    }

  } break;

  case LayoutKind::Struct: {
    auto *Struct = llvm::cast<dla::StructLayout>(L);

    std::string TypeName = getUniqueTypeNameForDecl(nullptr);
    clang::IdentifierInfo &TypeId = ClangCtx.Idents.get(TypeName);
    clang::TranslationUnitDecl &TUDecl = *ClangCtx.getTranslationUnitDecl();
    auto *StructDecl = clang::RecordDecl::Create(ClangCtx,
                                                 clang::TTK_Struct,
                                                 &TUDecl,
                                                 clang::SourceLocation{},
                                                 clang::SourceLocation{},
                                                 &TypeId,
                                                 nullptr);

    // We have to insert it in the mapping before looking at its child,
    // otherwise we risk to generate differenty types with duplicate names.
    Result = StructDecl;
    insertTypeMapping(L, Result);

    StructDecl->startDefinition();

    for (const auto &Group : llvm::enumerate(Struct->fields())) {
      // Create the type decl for the PointerLayout.
      const dla::Layout *PointedLayout = Group.value();
      TypeDeclOrQualType FTy = getOrCreateTypeFromLayout(PointedLayout,
                                                         ClangCtx,
                                                         LLVMCtx);
      QualType FieldTy = DeclCreator::getQualType(FTy);
      clang::TypeSourceInfo *TI = ClangCtx.CreateTypeSourceInfo(FieldTy);
      const std::string FieldName = std::string("field_")
                                    + std::to_string(Group.index());
      clang::IdentifierInfo &FieldId = ClangCtx.Idents.get(FieldName);
      auto *Field = clang::FieldDecl::Create(ClangCtx,
                                             StructDecl,
                                             clang::SourceLocation{},
                                             clang::SourceLocation{},
                                             &FieldId,
                                             FieldTy,
                                             TI,
                                             nullptr,
                                             /*Mutable*/ false,
                                             clang::ICIS_NoInit);
      StructDecl->addDecl(Field);
    }

    StructDecl->completeDefinition();

  } break;

  case LayoutKind::Union: {
    auto *Union = llvm::cast<dla::UnionLayout>(L);

    std::string TypeName = getUniqueTypeNameForDecl(nullptr);
    clang::IdentifierInfo &TypeId = ClangCtx.Idents.get(TypeName);
    clang::TranslationUnitDecl &TUDecl = *ClangCtx.getTranslationUnitDecl();
    auto *UnionDecl = clang::RecordDecl::Create(ClangCtx,
                                                clang::TTK_Union,
                                                &TUDecl,
                                                clang::SourceLocation{},
                                                clang::SourceLocation{},
                                                &TypeId,
                                                nullptr);

    // We have to insert it in the mapping before looking at its child,
    // otherwise we risk to generate differenty types with duplicate names.
    Result = UnionDecl;
    insertTypeMapping(L, Result);

    UnionDecl->startDefinition();

    for (const auto &Group : llvm::enumerate(Union->elements())) {
      // Create the type decl for the PointerLayout.
      const dla::Layout *PointedLayout = Group.value();
      TypeDeclOrQualType FTy = getOrCreateTypeFromLayout(PointedLayout,
                                                         ClangCtx,
                                                         LLVMCtx);
      QualType FieldTy = DeclCreator::getQualType(FTy);
      clang::TypeSourceInfo *TI = ClangCtx.CreateTypeSourceInfo(FieldTy);
      const std::string FieldName = std::string("field_")
                                    + std::to_string(Group.index());
      clang::IdentifierInfo &FieldId = ClangCtx.Idents.get(FieldName);
      auto *Field = clang::FieldDecl::Create(ClangCtx,
                                             UnionDecl,
                                             clang::SourceLocation{},
                                             clang::SourceLocation{},
                                             &FieldId,
                                             FieldTy,
                                             TI,
                                             nullptr,
                                             /*Mutable*/ false,
                                             clang::ICIS_NoInit);
      UnionDecl->addDecl(Field);
    }

    UnionDecl->completeDefinition();

  } break;

  default:
    revng_unreachable();
  }

  return Result;
}

// TODO: this should be a RecursiveCoroutine
DeclCreator::TypeDeclOrQualType
DeclCreator::getOrCreateTypeFromLayout(const dla::Layout *L,
                                       clang::ASTContext &ClangCtx,
                                       llvm::LLVMContext &LLVMCtx) {
  revng_assert(L);

  // First, look it up, to see if we've already computed it.
  llvm::Optional<TypeDeclOrQualType> CachedQTy = lookupType(L);
  if (CachedQTy.hasValue())
    return CachedQTy.getValue();

  // If we havent found it, create it, along with all the other types that might
  // be necessary to complete its declaration.
  TypeDeclOrQualType Result = createTypeFromLayout(L, ClangCtx, LLVMCtx);

  // Insert it into the mapping.
  insertTypeMapping(L, Result);
  return Result;
}

void DeclCreator::createTypeDeclsForFunctionPrototype(clang::ASTContext &Ctx,
                                                      const llvm::Function *F) {
  clang::TranslationUnitDecl &TUDecl = *Ctx.getTranslationUnitDecl();

  // Create return type
  getOrCreateType(F, Ctx, TUDecl);

  // Create argument types
  for (const llvm::Argument &A : F->args())
    getOrCreateType(&A, Ctx, TUDecl);
}
