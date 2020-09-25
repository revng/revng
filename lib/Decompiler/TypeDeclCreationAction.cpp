//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"

#include "revng/Support/Assert.h"

#include "revng-c/Decompiler/DLALayouts.h"

#include "DecompilationHelpers.h"
#include "IRASTTypeTranslation.h"
#include "Mangling.h"

namespace clang {
class TranslationUnitDecl;
} // end namespace clang

using clang::QualType;

QualType
DeclCreator::getOrCreateQualTypeFromLayout(const dla::Layout *L,
                                           clang::ASTContext &ClangCtx,
                                           llvm::LLVMContext &LLVMCtx) {
  revng_assert(L);

  if (auto It = LayoutQualTypes.find(L); It != LayoutQualTypes.end())
    return It->second;

  QualType Result = ClangCtx.VoidTy;

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
    bool IsPowerOf2 = (ByteSize & (ByteSize - 1)) == 0;
    revng_assert(IsPowerOf2);
    revng_assert(ByteSize <= 16);
    auto *IntTy = llvm::IntegerType::get(LLVMCtx, ByteSize * 8);
    Result = getOrCreateQualType(IntTy,
                                 nullptr,
                                 ClangCtx,
                                 *ClangCtx.getTranslationUnitDecl());
  } break;

  case LayoutKind::Array: {

    auto *Array = llvm::cast<dla::ArrayLayout>(L);
    dla::Layout *ElementLayout = Array->getElem();
    QualType ElemQualTy = getOrCreateQualTypeFromLayout(ElementLayout,
                                                        ClangCtx,
                                                        LLVMCtx);

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
    TypeDecls.push_back(StructDecl);

    StructDecl->startDefinition();

    for (const auto &Group : llvm::enumerate(Struct->fields())) {
      // Create the type decl for the PointerLayout.
      const dla::Layout *PointedLayout = Group.value();
      QualType FieldTy = getOrCreateQualTypeFromLayout(PointedLayout,
                                                       ClangCtx,
                                                       LLVMCtx);
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

    Result = QualType(StructDecl->getTypeForDecl(), 0);

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
    TypeDecls.push_back(UnionDecl);

    UnionDecl->startDefinition();

    for (const auto &Group : llvm::enumerate(Union->elements())) {
      // Create the type decl for the PointerLayout.
      const dla::Layout *PointedLayout = Group.value();
      QualType FieldTy = getOrCreateQualTypeFromLayout(PointedLayout,
                                                       ClangCtx,
                                                       LLVMCtx);
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

    Result = QualType(UnionDecl->getTypeForDecl(), 0);

  } break;

  default:
    revng_unreachable();
  }

  const auto &[_, New] = LayoutQualTypes.insert({ L, Result });
  revng_assert(New);
  return Result;
}

void DeclCreator::createTypeDeclsForFunctionPrototype(clang::ASTContext &Ctx,
                                                      const llvm::Function *F) {
  clang::TranslationUnitDecl &TUDecl = *Ctx.getTranslationUnitDecl();

  // Create return type
  getOrCreateValueQualType(F, Ctx, TUDecl);

  // Create argument types
  for (const llvm::Argument &A : F->args())
    getOrCreateValueQualType(&A, Ctx, TUDecl);
}
