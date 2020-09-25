//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"

#include "revng/Support/Assert.h"

#include "revng-c/Decompiler/DLALayouts.h"

#include "IRASTTypeTranslation.h"

#include "DLATypeSystem.h"
#include "Mangling.h"

std::string
DeclCreator::getUniqueTypeNameForDecl(const llvm::Value *NamingValue) const {
  std::string Result;

  if (NamingValue) {
    if (NamingValue->hasName()) {
      if (auto *F = llvm::dyn_cast<llvm::Function>(NamingValue)) {
        Result = makeCIdentifier(F->getName().str()) + std::string("_ret_type");
      } else if (auto *Arg = llvm::dyn_cast<llvm::Argument>(NamingValue)) {
        const llvm::Function *F = Arg->getParent();
        Result = makeCIdentifier(F->getName().str()) + "_arg_"
                 + makeCIdentifier(NamingValue->getName().str())
                 + std::string("_type");
      }
    }
    if (Result.empty()) {
      const llvm::Type *VTy = NamingValue->getType();
      if (const auto *StructTy = llvm::dyn_cast<llvm::StructType>(VTy))
        Result = makeCIdentifier(StructTy->getName());
    }
  }

  if (Result.empty())
    Result = std::string("type_") + std::to_string(TypeDecls.size());

  return Result;
}

clang::QualType DeclCreator::getOrCreateBoolQualType(clang::ASTContext &ASTCtx,
                                                     const llvm::Type *Ty) {

  clang::QualType Result;
  clang::TranslationUnitDecl *TUDecl = ASTCtx.getTranslationUnitDecl();
  const std::string BoolName = "bool";
  clang::IdentifierInfo &BoolId = ASTCtx.Idents.get(BoolName);
  clang::DeclarationName TypeName(&BoolId);
  bool Found = false;
  for (clang::Decl *D : TUDecl->lookup(TypeName)) {
    if (auto *Typedef = llvm::dyn_cast<clang::TypedefDecl>(D)) {
      const std::string Name = Typedef->getNameAsString();
      if (Name == "bool") {
        revng_assert(not Found);
        insertTypeMapping(Ty, Typedef);
        Result = ASTCtx.getTypedefType(Typedef);
        Found = true;
      }
    }
  }
  if (not Found) {
    // C99 actually defines '_Bool', while 'bool' is a MACRO, which expands
    // to '_Bool'. We cheat by injecting a 'typedef _Bool bool;' in the
    // translation unit we're handling, that has already been preprocessed
    clang::CanQualType BoolQType = ASTCtx.BoolTy;
    using TSInfo = clang::TypeSourceInfo;
    TSInfo *BoolTypeInfo = ASTCtx.getTrivialTypeSourceInfo(BoolQType);
    auto *BoolTypedefDecl = clang::TypedefDecl::Create(ASTCtx,
                                                       TUDecl,
                                                       {},
                                                       {},
                                                       &BoolId,
                                                       BoolTypeInfo);
    insertTypeMapping(Ty, BoolTypedefDecl);
    TUDecl->addDecl(BoolTypedefDecl);
    Result = ASTCtx.getTypedefType(BoolTypedefDecl);
    return Result;
    revng_abort("'_Bool' type not found!\n"
                "This should not happen since we compile as c99\n");
  }
  return Result;
}

clang::QualType DeclCreator::getOrCreateQualType(const llvm::Type *Ty,
                                                 const llvm::Value *NamingValue,
                                                 clang::ASTContext &ASTCtx,
                                                 clang::DeclContext &DeclCtx) {

  clang::QualType Result;
  switch (Ty->getTypeID()) {

  case llvm::Type::TypeID::IntegerTyID: {
    auto *LLVMIntType = llvm::cast<llvm::IntegerType>(Ty);
    unsigned BitWidth = LLVMIntType->getBitWidth();
    revng_assert(BitWidth == 1 or BitWidth == 8 or BitWidth == 16
                 or BitWidth == 32 or BitWidth == 64 or BitWidth == 128);

    clang::TranslationUnitDecl *TUDecl = ASTCtx.getTranslationUnitDecl();
    revng_assert(TUDecl->isLookupContext());

    switch (BitWidth) {
    case 1: {
      Result = getOrCreateBoolQualType(ASTCtx, Ty);
    } break;
    case 16: {
      const std::string UInt16Name = "uint16_t";
      clang::IdentifierInfo &Id = ASTCtx.Idents.get(UInt16Name);
      clang::DeclarationName TypeName(&Id);
      for (clang::Decl *D : TUDecl->lookup(TypeName))
        if (auto *Typedef = llvm::dyn_cast<clang::TypedefDecl>(D))
          return ASTCtx.getTypedefType(Typedef);
      revng_abort("'uint16_t' type not found!\n"
                  "This should not happen since we '#include <stdint.h>'\n"
                  "Please make sure you have installed the header\n");
    } break;
    case 32: {
      const std::string UInt32Name = "uint32_t";
      clang::IdentifierInfo &Id = ASTCtx.Idents.get(UInt32Name);
      clang::DeclarationName TypeName(&Id);
      for (clang::Decl *D : TUDecl->lookup(TypeName))
        if (auto *Typedef = llvm::dyn_cast<clang::TypedefDecl>(D))
          return ASTCtx.getTypedefType(Typedef);
      revng_abort("'uint32_t' type not found!\n"
                  "This should not happen since we '#include <stdint.h>'\n"
                  "Please make sure you have installed the header\n");
    } break;
    case 64: {
      const std::string UInt64Name = "uint64_t";
      clang::IdentifierInfo &Id = ASTCtx.Idents.get(UInt64Name);
      clang::DeclarationName TypeName(&Id);
      for (clang::Decl *D : TUDecl->lookup(TypeName))
        if (auto *Typedef = llvm::dyn_cast<clang::TypedefDecl>(D))
          return ASTCtx.getTypedefType(Typedef);
      revng_abort("'uint64_t' type not found!\n"
                  "This should not happen since we '#include <stdint.h>'\n"
                  "Please make sure you have installed the header\n");
    } break;
    case 8:
      // use char for strict aliasing reasons
    case 128:
      // use builtin clang types (__int128_t and __uint128_t), because C99 does
      // not require to have 128 bit integer types
      Result = ASTCtx.getIntTypeForBitwidth(BitWidth, /* Signed */ false);
      break;
    default:
      revng_abort("unexpected integer size");
    }
  } break;

  case llvm::Type::TypeID::PointerTyID: {
    const llvm::PointerType *PtrTy = llvm::cast<llvm::PointerType>(Ty);
    const llvm::Type *PointeeTy = PtrTy->getElementType();
    clang::QualType PointeeType = getOrCreateQualType(PointeeTy,
                                                      nullptr,
                                                      ASTCtx,
                                                      DeclCtx);
    Result = ASTCtx.getPointerType(PointeeType);
  } break;

  case llvm::Type::TypeID::VoidTyID:
  case llvm::Type::TypeID::MetadataTyID: {
    Result = ASTCtx.VoidTy;
  } break;

  case llvm::Type::TypeID::StructTyID: {
    if (auto *TDecl = getTypeDeclOrNull(Ty)) {
      Result = clang::QualType(TDecl->getTypeForDecl(), 0);
      break;
    }

    std::string TypeName = getUniqueTypeNameForDecl(NamingValue);
    clang::IdentifierInfo &TypeId = ASTCtx.Idents.get(TypeName);
    auto *Struct = clang::RecordDecl::Create(ASTCtx,
                                             clang::TTK_Struct,
                                             &DeclCtx,
                                             clang::SourceLocation{},
                                             clang::SourceLocation{},
                                             &TypeId,
                                             nullptr);
    insertTypeMapping(Ty, Struct);
    Struct->startDefinition();
    const auto *StructTy = llvm::cast<llvm::StructType>(Ty);
    for (auto &Group : llvm::enumerate(StructTy->elements())) {
      llvm::Type *FieldTy = Group.value();
      // HACK: Handle the type of the `@env` global variable, which we simply
      //       cast to a `void` `nullptr` type.
      if (FieldTy->getTypeID() == llvm::Type::TypeID::ArrayTyID) {
        Result = ASTCtx.VoidTy;
        break;
      }
      clang::QualType QFieldTy = getOrCreateQualType(Group.value(),
                                                     nullptr,
                                                     ASTCtx,
                                                     DeclCtx);
      clang::TypeSourceInfo *TI = ASTCtx.CreateTypeSourceInfo(QFieldTy);
      const std::string FieldName = std::string("field_")
                                    + std::to_string(Group.index());
      clang::IdentifierInfo &FieldId = ASTCtx.Idents.get(FieldName);
      auto *Field = clang::FieldDecl::Create(ASTCtx,
                                             Struct,
                                             clang::SourceLocation{},
                                             clang::SourceLocation{},
                                             &FieldId,
                                             QFieldTy,
                                             TI,
                                             nullptr,
                                             /*Mutable*/ false,
                                             clang::ICIS_NoInit);
      Struct->addDecl(Field);
    }
    Struct->completeDefinition();
    Result = clang::QualType(Struct->getTypeForDecl(), 0);
  } break;
  case llvm::Type::TypeID::ArrayTyID:
  case llvm::Type::TypeID::FunctionTyID:
  case llvm::Type::TypeID::VectorTyID:
  case llvm::Type::TypeID::LabelTyID:
  case llvm::Type::TypeID::X86_MMXTyID:
  case llvm::Type::TypeID::TokenTyID:
  case llvm::Type::TypeID::HalfTyID:
  case llvm::Type::TypeID::FloatTyID:
  case llvm::Type::TypeID::DoubleTyID:
  case llvm::Type::TypeID::X86_FP80TyID:
  case llvm::Type::TypeID::FP128TyID:
  case llvm::Type::TypeID::PPC_FP128TyID:
    revng_abort("unsupported type");
  }
  revng_assert(not Result.isNull());
  return Result;
}

llvm::SmallVector<const dla::Layout *, 16>
DeclCreator::getPointedLayouts(const llvm::Value *V) const {

  revng_assert(V);

  llvm::SmallVector<const dla::Layout *, 16> Result;

  if (not ValueLayouts)
    return Result;

  // If we have a ValueLayouts map, it means that the DLA analysis was
  // executed. We want to customize V's type in C according to the
  // layouts computed by the DLA.
  auto FItBegin = ValueLayouts->lower_bound(dla::LayoutTypePtr(V, 0));
  auto FItEnd = ValueLayouts->upper_bound(dla::LayoutTypePtr(V));

  // If there is a range of ValueLayouts that are indexed by V, we have type
  // information coming from the DLA about where V points.
  if (FItBegin != FItEnd) {

    if (std::next(FItBegin) == FItEnd) {

      // If we only find a single entry in ValueLayouts associated with V,
      // we expect this to be a scalar type.
      auto FieldId = FItBegin->first.fieldNum();
      revng_assert(FieldId == dla::LayoutTypePtr::fieldNumNone);

      // V has a scalar type, which is a pointer to the pointed Layout.
      Result.push_back(FItBegin->second);

    } else {

      // If we find more than one entry in ValueLayouts associated with V, it
      // means that V ha either a struct type, or it is a function that
      // returns a struct type.
      // Each entry in ValueLayouts corresponds to a field of a struct.
      // The field of a struct has a pointer type and points to a Layout that
      // is described by the mapped value in ValueLayouts.

      // TODO: we might be losing fields at the end of the struct.
      auto LastFieldId = std::prev(FItBegin)->first.fieldNum();
      revng_assert(LastFieldId != dla::LayoutTypePtr::fieldNumNone);
      for (decltype(LastFieldId) FieldId = 0; FieldId <= LastFieldId;
           ++FieldId) {

        auto CurrId = FItBegin->first.fieldNum();
        if (CurrId == FieldId) {
          Result.push_back(FItBegin->second);
          ++FItBegin;
        } else {
          Result.push_back(nullptr);
        }
      }
    }
  }
  return Result;
}

llvm::Optional<clang::QualType>
DeclCreator::getOrCreateDLAQualType(const llvm::Value *V,
                                    clang::ASTContext &ASTCtx,
                                    clang::DeclContext &DeclCtx) {
  revng_assert(V);

  llvm::Optional<clang::QualType> Result;

  if (auto It = ValueQualTypes.find(V); It != ValueQualTypes.end()) {

    return Result = It->second;

  } else if (ValueLayouts) {

    auto PointedLayouts = getPointedLayouts(V);

    // If there is a range of ValueLayouts that are indexed by V, we have type
    // information coming from the DLA about where V points.
    if (not PointedLayouts.empty()) {

      llvm::LLVMContext &LLVMCtx = V->getContext();

      if (PointedLayouts.size() == 1) {
        // V has a scalar type, which is a pointer to the PointedLayout.
        const dla::Layout *PointedLayout = PointedLayouts.front();
        revng_assert(PointedLayout);
        clang::QualType LayoutTy = getOrCreateQualTypeFromLayout(PointedLayout,
                                                                 ASTCtx,
                                                                 LLVMCtx);
        Result = ASTCtx.getPointerType(LayoutTy);

      } else {

        // If we find more than one entry in ValueLayouts associated with V, it
        // means that V ha either a struct type, or it is a function that
        // returns a struct type.
        // Each entry in ValueLayouts corresponds to a field of a struct.
        // The field of a struct has a pointer type and points to a Layout that
        // is described by the mapped value in ValueLayouts.

        // Creat the struct
        std::string TypeName = getUniqueTypeNameForDecl(nullptr);
        clang::IdentifierInfo &TypeId = ASTCtx.Idents.get(TypeName);
        auto *Struct = clang::RecordDecl::Create(ASTCtx,
                                                 clang::TTK_Struct,
                                                 &DeclCtx,
                                                 clang::SourceLocation{},
                                                 clang::SourceLocation{},
                                                 &TypeId,
                                                 nullptr);
        TypeDecls.push_back(Struct);

        // Define the fields.
        Struct->startDefinition();

        for (const auto &Group : llvm::enumerate(PointedLayouts)) {
          // Create the type decl for the PointerLayout.
          clang::QualType LayoutTy;
          if (const dla::Layout *PointedLayout = Group.value()) {
            LayoutTy = getOrCreateQualTypeFromLayout(PointedLayout,
                                                     ASTCtx,
                                                     LLVMCtx);
          } else {
            // If we don't have information about the type of this field from
            // the DLA we just say it's a pointer to void.
            LayoutTy = ASTCtx.VoidTy;
          }
          clang::QualType FieldPtrTy = ASTCtx.getPointerType(LayoutTy);

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
          Struct->addDecl(Field);
        }

        Struct->completeDefinition();

        Result = clang::QualType(Struct->getTypeForDecl(), 0);
      }
    }
  }

  // If we found a DLA type add it to the map and we're done.
  if (Result.hasValue()) {
    const auto &[_, New] = ValueQualTypes.insert({ V, Result.getValue() });
    revng_assert(New);
  }

  return Result;
}

clang::QualType
DeclCreator::getOrCreateValueQualType(const llvm::Value *V,
                                      clang::ASTContext &ASTCtx,
                                      clang::DeclContext &DeclCtx) {
  revng_assert(V);

  llvm::Optional<clang::QualType> Result = getOrCreateDLAQualType(V,
                                                                  ASTCtx,
                                                                  DeclCtx);
  if (Result.hasValue())
    return Result.getValue();

  const auto *VType = V->getType();
  if (const auto *F = dyn_cast<llvm::Function>(V)) {
    VType = F->getReturnType();
  } else if (const auto *G = dyn_cast<llvm::GlobalVariable>(V)) {
    const auto *GlobalPtrTy = llvm::cast<llvm::PointerType>(G->getType());
    VType = GlobalPtrTy->getElementType();
  }

  Result = getOrCreateQualType(VType, V, ASTCtx, DeclCtx);

  const auto &[_, New] = ValueQualTypes.insert({ V, Result.getValue() });
  revng_assert(New);
  return Result.getValue();
}
