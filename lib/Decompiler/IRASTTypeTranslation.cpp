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
    Result = std::string("type_") + std::to_string(Types.size());

  return Result;
}

DeclCreator::TypeDeclOrQualType
DeclCreator::createBoolType(clang::ASTContext &ASTCtx) {
  clang::TypedefDecl *BoolTypedefDecl = nullptr;

  clang::TranslationUnitDecl *TUDecl = ASTCtx.getTranslationUnitDecl();
  const std::string BoolName = "bool";
  clang::IdentifierInfo &BoolId = ASTCtx.Idents.get(BoolName);
  clang::DeclarationName TypeName(&BoolId);

  // First, lookup for a typedef called 'bool', if we find it we're do
  for (clang::Decl *D : TUDecl->lookup(TypeName)) {
    if (auto *Typedef = llvm::dyn_cast<clang::TypedefDecl>(D)) {
      // Assert that we only find one typedef called 'bool'
      revng_assert(Typedef->getNameAsString() != BoolName
                   or not BoolTypedefDecl);
      BoolTypedefDecl = Typedef;
    }
  }

  // If we haven't found the typedef, create it ourselves
  if (not BoolTypedefDecl) {
    // C99 actually defines '_Bool', while 'bool' is a MACRO, which expands
    // to '_Bool'. We cheat by injecting a 'typedef _Bool bool;' in the
    // translation unit we're handling, that has already been preprocessed
    clang::CanQualType BoolQType = ASTCtx.BoolTy;
    revng_assert(not BoolQType.isNull());
    using TSInfo = clang::TypeSourceInfo;
    TSInfo *BoolTypeInfo = ASTCtx.getTrivialTypeSourceInfo(BoolQType);
    BoolTypedefDecl = clang::TypedefDecl::Create(ASTCtx,
                                                 TUDecl,
                                                 {},
                                                 {},
                                                 &BoolId,
                                                 BoolTypeInfo);

    revng_assert(BoolTypedefDecl,
                 "'_Bool' type not found!\n"
                 "This should not happen since we compile as c99\n");

    TUDecl->addDecl(BoolTypedefDecl);
  }

  return ASTCtx.getTypedefType(BoolTypedefDecl);
}

DeclCreator::TypeDeclOrQualType
DeclCreator::getOrCreateBoolType(clang::ASTContext &ASTCtx,
                                 const llvm::Type *Ty) {
  // First, look it up, to see if we've already computed it.
  llvm::Optional<TypeDeclOrQualType> CachedQTy = lookupType(Ty);
  if (CachedQTy.hasValue())
    return CachedQTy.getValue();

  // Otherwise, create it right now.
  TypeDeclOrQualType BoolTypedefDecl = createBoolType(ASTCtx);

  // And add it to the map.
  insertTypeMapping</* NullableTy */ true>(Ty, BoolTypedefDecl);
  return BoolTypedefDecl;
}

DeclCreator::TypeDeclOrQualType
DeclCreator::createType(const llvm::Type *Ty,
                        const llvm::Value *NamingValue,
                        clang::ASTContext &ASTCtx,
                        clang::DeclContext &DeclCtx) {

  llvm::Optional<TypeDeclOrQualType> Result = llvm::None;

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
      Result = getOrCreateBoolType(ASTCtx, Ty);
    } break;

    case 16: {
      const std::string UInt16Name = "uint16_t";
      clang::IdentifierInfo &Id = ASTCtx.Idents.get(UInt16Name);
      clang::DeclarationName TypeName(&Id);
      for (clang::Decl *D : TUDecl->lookup(TypeName)) {
        if (auto *Typedef = llvm::dyn_cast<clang::TypedefDecl>(D)) {
          revng_assert(not Result.hasValue());
          Result = ASTCtx.getTypedefType(Typedef);
        }
      }

      revng_assert(Result.hasValue(),
                   "'uint16_t' type not found!\n"
                   "This should not happen since we '#include <stdint.h>'\n"
                   "Please make sure you have installed the header\n");
    } break;

    case 32: {
      const std::string UInt32Name = "uint32_t";
      clang::IdentifierInfo &Id = ASTCtx.Idents.get(UInt32Name);
      clang::DeclarationName TypeName(&Id);
      for (clang::Decl *D : TUDecl->lookup(TypeName)) {
        if (auto *Typedef = llvm::dyn_cast<clang::TypedefDecl>(D)) {
          revng_assert(not Result.hasValue());
          Result = ASTCtx.getTypedefType(Typedef);
        }
      }

      revng_assert(Result.hasValue(),
                   "'uint32_t' type not found!\n"
                   "This should not happen since we '#include <stdint.h>'\n"
                   "Please make sure you have installed the header\n");
    } break;

    case 64: {
      const std::string UInt64Name = "uint64_t";
      clang::IdentifierInfo &Id = ASTCtx.Idents.get(UInt64Name);
      clang::DeclarationName TypeName(&Id);
      for (clang::Decl *D : TUDecl->lookup(TypeName)) {
        if (auto *Typedef = llvm::dyn_cast<clang::TypedefDecl>(D)) {
          revng_assert(not Result.hasValue());
          Result = ASTCtx.getTypedefType(Typedef);
        }
      }

      revng_assert(Result.hasValue(),
                   "'uint64_t' type not found!\n"
                   "This should not happen since we '#include <stdint.h>'\n"
                   "Please make sure you have installed the header\n");
    } break;

    case 8:
      // use char for strict aliasing reasons
      [[fallthrough]];
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
    TypeDeclOrQualType PointeeType = getOrCreateType(PointeeTy,
                                                     nullptr,
                                                     ASTCtx,
                                                     DeclCtx);
    Result = ASTCtx.getPointerType(getQualType(PointeeType));
  } break;

  case llvm::Type::TypeID::VoidTyID:
  case llvm::Type::TypeID::MetadataTyID: {
    Result = ASTCtx.VoidTy;
  } break;

  case llvm::Type::TypeID::StructTyID: {

    std::string TypeName = getUniqueTypeNameForDecl(NamingValue);
    clang::IdentifierInfo &TypeId = ASTCtx.Idents.get(TypeName);
    auto *Struct = clang::RecordDecl::Create(ASTCtx,
                                             clang::TTK_Struct,
                                             &DeclCtx,
                                             clang::SourceLocation{},
                                             clang::SourceLocation{},
                                             &TypeId,
                                             nullptr);
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
      TypeDeclOrQualType FTy = getOrCreateType(Group.value(),
                                               nullptr,
                                               ASTCtx,
                                               DeclCtx);
      clang::QualType QFieldTy = DeclCreator::getQualType(FTy);
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
    Result = Struct;
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

  revng_assert(Result.hasValue());
  revng_assert(not DeclCreator::getQualType(Result.getValue()).isNull());
  return Result.getValue();
}

DeclCreator::TypeDeclOrQualType
DeclCreator::getOrCreateType(const llvm::Type *Ty,
                             const llvm::Value *NamingValue,
                             clang::ASTContext &ASTCtx,
                             clang::DeclContext &DeclCtx) {

  // First, look it up, to see if we've already computed it.
  llvm::Optional<TypeDeclOrQualType> CachedQTy = lookupType(Ty);
  if (CachedQTy.hasValue())
    return CachedQTy.getValue();

  // Otherwise, create it
  TypeDeclOrQualType Result = createType(Ty, NamingValue, ASTCtx, DeclCtx);

  // Then insert it in the type mapping and return it.
  insertTypeMapping(Ty, Result);
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

llvm::Optional<DeclCreator::TypeDeclOrQualType>
DeclCreator::getOrCreateDLAType(const llvm::Value *V,
                                clang::ASTContext &ASTCtx,
                                clang::DeclContext &DeclCtx) {
  revng_assert(V);

  // First, look it up, to see if we've already computed it.
  llvm::Optional<TypeDeclOrQualType> CachedQTy = lookupType(V);
  if (CachedQTy.hasValue())
    return CachedQTy.getValue();

  llvm::Optional<TypeDeclOrQualType> Result = llvm::None;

  if (ValueLayouts) {

    auto PointedLayouts = getPointedLayouts(V);

    // If there is a range of ValueLayouts that are indexed by V, we have type
    // information coming from the DLA about where V points.
    if (not PointedLayouts.empty()) {

      llvm::LLVMContext &LLVMCtx = V->getContext();

      if (PointedLayouts.size() == 1) {
        // V has a scalar type, which is a pointer to the PointedLayout.
        const dla::Layout *PointedLayout = PointedLayouts.front();
        revng_assert(PointedLayout);
        TypeDeclOrQualType LTy = getOrCreateTypeFromLayout(PointedLayout,
                                                           ASTCtx,
                                                           LLVMCtx);
        Result = ASTCtx.getPointerType(DeclCreator::getQualType(LTy));

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
        // Define the fields.
        Struct->startDefinition();

        for (const auto &Group : llvm::enumerate(PointedLayouts)) {
          // Create the type decl for the PointerLayout.
          TypeDeclOrQualType LayoutTy;
          if (const dla::Layout *PointedLayout = Group.value()) {
            LayoutTy = getOrCreateTypeFromLayout(PointedLayout,
                                                 ASTCtx,
                                                 LLVMCtx);
          } else {
            // If we don't have information about the type of this field from
            // the DLA we just say it's a pointer to void.
            LayoutTy = ASTCtx.VoidTy;
          }

          clang::QualType LayoutQualTy = DeclCreator::getQualType(LayoutTy);
          clang::QualType FieldPtrTy = ASTCtx.getPointerType(LayoutQualTy);

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

        Result = Struct;
      }
    }
  }

  if (Result.hasValue())
    insertTypeMapping(V, Result.getValue());

  return Result;
}

DeclCreator::TypeDeclOrQualType
DeclCreator::getOrCreateType(const llvm::Value *V,
                             clang::ASTContext &ASTCtx,
                             clang::DeclContext &DeclCtx) {
  revng_assert(V);

  llvm::Optional<TypeDeclOrQualType> Result = getOrCreateDLAType(V,
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

  return getOrCreateType(VType, V, ASTCtx, DeclCtx);
}
