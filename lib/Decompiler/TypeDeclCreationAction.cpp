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

  // FIXME: here, build the type decl for the C type associated to
  // layout.
  using LayoutKind = dla::Layout::LayoutKind;
  revng_assert(L->size());

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

    std::string TypeName = getUniqueTypeNameForDecl();
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
      FieldDecls[StructDecl].push_back(Field);
      StructDecl->addDecl(Field);
    }

    StructDecl->completeDefinition();

    Result = QualType(StructDecl->getTypeForDecl(), 0);

  } break;

  case LayoutKind::Union: {
    auto *Union = llvm::cast<dla::UnionLayout>(L);

    std::string TypeName = getUniqueTypeNameForDecl();
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
      FieldDecls[UnionDecl].push_back(Field);
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

QualType DeclCreator::getOrCreateFunctionRetType(llvm::Function *F,
                                                 clang::ASTContext &ASTCtx,
                                                 clang::DeclContext &DeclCtx) {
  QualType Result;
  // If we have a ValueLayouts map, it means that the DLA analysis was
  // executed. We want to customize F's return type in C according to the
  // layouts computed by the DLA.
  if (nullptr != ValueLayouts) {
    if (auto It = FunctionRetTypes.find(F); It != FunctionRetTypes.end())
      return It->second;

    auto FItBegin = ValueLayouts->lower_bound(dla::LayoutTypePtr(F, 0));
    auto FItEnd = ValueLayouts->upper_bound(dla::LayoutTypePtr(F));
    // If there is a range of ValueLayouts that are indexed by F, they
    // represent the return value of F.
    if (FItBegin != FItEnd) {
      llvm::LLVMContext &LLVMCtx = F->getContext();
      if (std::next(FItBegin) == FItEnd) {
        // F returns a scalar type, which is a pointer to the PointedLayout.
        dla::Layout *PointedLayout = FItBegin->second;
        revng_assert(PointedLayout);

        QualType LayoutTy = getOrCreateQualTypeFromLayout(PointedLayout,
                                                          ASTCtx,
                                                          LLVMCtx);
        Result = ASTCtx.getPointerType(LayoutTy);
      } else {

        revng_assert(F->hasName());
        std::string TypeName = makeCIdentifier(F->getName()) + "_return_t";

        clang::IdentifierInfo &TypeId = ASTCtx.Idents.get(TypeName);
        auto *Struct = clang::RecordDecl::Create(ASTCtx,
                                                 clang::TTK_Struct,
                                                 &DeclCtx,
                                                 clang::SourceLocation{},
                                                 clang::SourceLocation{},
                                                 &TypeId,
                                                 nullptr);
        TypeDecls.push_back(Struct);

        Struct->startDefinition();

        auto LayoutMapRange = llvm::make_range(FItBegin, FItEnd);
        for (const auto &Group : llvm::enumerate(LayoutMapRange)) {
          // Create the type decl for the PointerLayout.
          dla::Layout *PointedLayout = Group.value().second;
          QualType LayoutTy = getOrCreateQualTypeFromLayout(PointedLayout,
                                                            ASTCtx,
                                                            LLVMCtx);
          QualType FieldPtrTy = ASTCtx.getPointerType(LayoutTy);

          clang::TypeSourceInfo *TI = ASTCtx.CreateTypeSourceInfo(FieldPtrTy);
          const std::string FieldName = std::string("field_")
                                        + std::to_string(Group.index());
          clang::IdentifierInfo &FieldId = ASTCtx.Idents.get(FieldName);
          auto *Field = clang::FieldDecl::Create(ASTCtx,
                                                 Struct,
                                                 clang::SourceLocation{},
                                                 clang::SourceLocation{},
                                                 &FieldId,
                                                 FieldPtrTy,
                                                 TI,
                                                 nullptr,
                                                 /*Mutable*/ false,
                                                 clang::ICIS_NoInit);
          FieldDecls[Struct].push_back(Field);
          Struct->addDecl(Field);
        }

        Struct->completeDefinition();

        Result = QualType(Struct->getTypeForDecl(), 0);
      }

      const auto &[_, New] = FunctionRetTypes.insert({ F, Result });
      revng_assert(New);
      return Result;
    }
  }

  const llvm::FunctionType *FType = F->getFunctionType();
  auto *RetType = FType->getReturnType();
  Result = getOrCreateQualType(RetType, F, ASTCtx, DeclCtx);

  const auto &[_, New] = FunctionRetTypes.insert({ F, Result });
  revng_assert(New);
  return Result;
}

QualType DeclCreator::getOrCreateArgumentType(llvm::Argument *A,
                                              clang::ASTContext &ASTCtx,
                                              clang::DeclContext &DeclCtx) {
  // For now we don't handle function prototypes with struct arguments.
  // In principle, we expect to never need it, because in assembly arguments
  // are passed to functions by means of registers, that in the decompiled C
  // code will become scalar type.
  revng_assert(not isa<llvm::StructType>(A->getType()));
  revng_assert(not A->getType()->isVoidTy());

  QualType Result;
  // If we have a ValueLayouts map, it means that the DLA analysis was
  // executed. We want to customize A's return type in C according to the
  // layouts computed by the DLA.
  if (nullptr != ValueLayouts) {
    if (auto It = ArgumentTypes.find(A); It != ArgumentTypes.end()) {
      Result = It->second;
      return Result;
    }

    if (auto It = ValueLayouts->find(dla::LayoutTypePtr(A));
        It != ValueLayouts->end()) {
      revng_assert(It->first.fieldNum() == dla::LayoutTypePtr::fieldNumNone);
      llvm::LLVMContext &LLVMCtx = A->getContext();
      // A must be a scalar type, which is a pointer to the PointedLayout.
      dla::Layout *PointedLayout = It->second;
      revng_assert(PointedLayout);

      QualType LayoutTy = getOrCreateQualTypeFromLayout(PointedLayout,
                                                        ASTCtx,
                                                        LLVMCtx);
      Result = ASTCtx.getPointerType(LayoutTy);

      const auto &[_, New] = ArgumentTypes.insert({ A, Result });
      revng_assert(New);
      return Result;
    }
  }

  Result = getOrCreateQualType(A->getType(), A, ASTCtx, DeclCtx);

  const auto &[_, New] = ArgumentTypes.insert({ A, Result });
  revng_assert(New);
  return Result;
}

void DeclCreator::createTypeDeclsForFunctionPrototype(clang::ASTContext &Ctx,
                                                      llvm::Function *F) {
  clang::TranslationUnitDecl &TUDecl = *Ctx.getTranslationUnitDecl();

  // Create return type
  getOrCreateFunctionRetType(F, Ctx, TUDecl);

  // Create argument types
  for (llvm::Argument &A : F->args())
    getOrCreateArgumentType(&A, Ctx, TUDecl);
}
