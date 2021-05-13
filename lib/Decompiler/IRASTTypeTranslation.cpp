//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <bit>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
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
        Result = makeCIdentifier(StructTy->getName().str());
    }
  }

  if (Result.empty())
    Result = std::string("type_") + std::to_string(Types.size());

  return Result;
}

DeclCreator::TypeDeclOrQualType
DeclCreator::getOrCreateBoolType(clang::ASTContext &ASTCtx,
                                 const llvm::Type *Ty) {

  // First, look it up, to see if we've already computed it.
  llvm::Optional<TypeDeclOrQualType> Result = lookupType(Ty);
  if (Result.hasValue())
    return Result.getValue();

  // Otherwise, create it right now.

  clang::TranslationUnitDecl *TUDecl = ASTCtx.getTranslationUnitDecl();
  clang::IdentifierInfo *BoolId = &ASTCtx.Idents.get("bool");
  clang::DeclarationName BoolTypeName(BoolId);

  // First, lookup for a typedef called 'bool', if we find it we're done
  auto LookupResult = TUDecl->lookup(BoolTypeName);
  revng_assert(LookupResult.empty() or LookupResult.size() == 1);

  clang::QualType BoolType;
  if (not LookupResult.empty()) {

    auto *BoolTypedef = llvm::cast<clang::TypedefDecl>(LookupResult.front());
    BoolType = ASTCtx.getTypedefType(BoolTypedef);

  } else {

    // If we haven't found the typedef, create it ourselves
    //
    // C99 actually defines '_Bool', while 'bool' is a MACRO, which expands
    // to '_Bool'. We cheat by injecting a 'typedef _Bool bool;' in the
    // translation unit we're handling, that has already been preprocessed
    clang::CanQualType BoolQType = ASTCtx.BoolTy;
    revng_assert(not BoolQType.isNull());
    using TSInfo = clang::TypeSourceInfo;
    TSInfo *BoolTypeInfo = ASTCtx.getTrivialTypeSourceInfo(BoolQType);
    auto *BoolTypedef = clang::TypedefDecl::Create(ASTCtx,
                                                   TUDecl,
                                                   {},
                                                   {},
                                                   BoolId,
                                                   BoolTypeInfo);
    TUDecl->addDecl(BoolTypedef);
    BoolType = ASTCtx.getTypedefType(BoolTypedef);
  }

  // Add it to the map.
  insertTypeMapping</* NullableTy */ true>(Ty, BoolType);
  return BoolType;
}

DeclCreator::TypeDeclOrQualType
DeclCreator::getOrCreateType(const llvm::Type *Ty,
                             const llvm::Value *NamingValue,
                             clang::ASTContext &ASTCtx,
                             clang::DeclContext &DeclCtx,
                             bool AllowArbitraryBitSize) {

  // First, look it up, to see if we've already computed it.
  llvm::Optional<TypeDeclOrQualType> CachedQTy = lookupType(Ty);
  if (CachedQTy.hasValue())
    return CachedQTy.getValue();

  llvm::Optional<TypeDeclOrQualType> Result = llvm::None;

  switch (Ty->getTypeID()) {

  case llvm::Type::TypeID::IntegerTyID: {
    auto *LLVMIntType = llvm::cast<llvm::IntegerType>(Ty);

    uint64_t BitWidth = LLVMIntType->getBitWidth();
    if (AllowArbitraryBitSize) {
      if (not std::has_single_bit(BitWidth))
        BitWidth = std::bit_ceil(BitWidth);
    } else {
      revng_assert(std::has_single_bit(BitWidth));
    }

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
    TypeDeclOrQualType PointeeType = PointeeTy->isFunctionTy() ?
                                       ASTCtx.VoidTy :
                                       getOrCreateType(PointeeTy,
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

    // We have to insert it in the mapping before looking at its child,
    // otherwise we risk to generate differenty types with duplicate names.
    Result = Struct;
    insertTypeMapping(Ty, Result.getValue());

    Struct->startDefinition();
    const auto *StructTy = llvm::cast<llvm::StructType>(Ty);
    for (auto &Group : llvm::enumerate(StructTy->elements())) {
      TypeDeclOrQualType FTy = getOrCreateType(Group.value(),
                                               nullptr,
                                               ASTCtx,
                                               DeclCtx,
                                               true);
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

      // Detect fields with sizes different from a power of 2, that need to be
      // rendered in C with bitfields
      auto *IntFTy = llvm::dyn_cast<llvm::IntegerType>(Group.value());
      if (IntFTy and not std::has_single_bit(IntFTy->getBitWidth())) {
        llvm::APInt FieldSize(32 /* bits */, IntFTy->getBitWidth());

        using clang::IntegerLiteral;
        auto *BitFieldLiteral = IntegerLiteral::Create(ASTCtx,
                                                       FieldSize,
                                                       ASTCtx.IntTy,
                                                       clang::SourceLocation{});

        using clang::ConstantExpr;
        auto *BitFieldSize = ConstantExpr::Create(ASTCtx, BitFieldLiteral);

        Field->setBitWidth(BitFieldSize);
      }

      Struct->addDecl(Field);
    }
    Struct->completeDefinition();

  } break;

  case llvm::Type::TypeID::ArrayTyID: {
    const auto *ArrayTy = llvm::cast<llvm::ArrayType>(Ty);
    uint64_t NElem = ArrayTy->getNumElements();
    const llvm::Type *ElemTy = ArrayTy->getElementType();
    auto ArraySizeKind = clang::ArrayType::ArraySizeModifier::Normal;
    TypeDeclOrQualType ElemDeclOrQualTy = getOrCreateType(ElemTy,
                                                          nullptr,
                                                          ASTCtx,
                                                          DeclCtx);

    revng_assert(not DeclCreator::getQualType(ElemDeclOrQualTy).isNull());
    clang::QualType ElemQualTy = DeclCreator::getQualType(ElemDeclOrQualTy);
    llvm::APInt ArraySize(64 /* bits */, NElem);
    Result = ASTCtx.getConstantArrayType(ElemQualTy,
                                         ArraySize,
                                         nullptr,
                                         ArraySizeKind,
                                         0);

  } break;

  case llvm::Type::TypeID::FloatTyID: {
    Result = ASTCtx.FloatTy;
  } break;

  case llvm::Type::TypeID::DoubleTyID: {
    Result = ASTCtx.DoubleTy;
  } break;

  case llvm::Type::TypeID::FunctionTyID:
  case llvm::Type::TypeID::FixedVectorTyID:
  case llvm::Type::TypeID::ScalableVectorTyID:
  case llvm::Type::TypeID::LabelTyID:
  case llvm::Type::TypeID::TokenTyID:
  case llvm::Type::TypeID::HalfTyID:
  case llvm::Type::TypeID::X86_FP80TyID:
  case llvm::Type::TypeID::X86_MMXTyID:
  case llvm::Type::TypeID::X86_AMXTyID:
  case llvm::Type::TypeID::BFloatTyID:
  case llvm::Type::TypeID::FP128TyID:
  case llvm::Type::TypeID::PPC_FP128TyID:
    revng_abort("unsupported type");
  }

  revng_assert(Result.hasValue());
  revng_assert(not DeclCreator::getQualType(Result.getValue()).isNull());

  // Then insert it in the type mapping and return it.
  TypeDeclOrQualType ResultVal = Result.getValue();
  insertTypeMapping(Ty, ResultVal);
  return ResultVal;
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

    if (std::next(FItBegin) == FItEnd
        and FItBegin->first.fieldNum() == dla::LayoutTypePtr::fieldNumNone) {

      // V has a scalar type, which is a pointer to the pointed Layout.
      Result.push_back(FItBegin->second);

    } else {

      // If we find more than one entry in ValueLayouts associated with V, it
      // means that V either has a struct type, or it is a function that
      // returns a struct type.
      // Each entry in ValueLayouts corresponds to a field of a struct.
      // The field of a struct has a pointer type and points to a Layout that
      // is described by the mapped value in ValueLayouts.

      llvm::Type *VTy = V->getType();
      if (const auto *Fun = dyn_cast<llvm::Function>(V))
        VTy = Fun->getReturnType();

      auto NumFields = llvm::cast<llvm::StructType>(VTy)->getNumElements();
      revng_assert(NumFields != dla::LayoutTypePtr::fieldNumNone);
      for (decltype(NumFields) FieldId = 0; FieldId < NumFields; ++FieldId) {

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
  llvm::Optional<TypeDeclOrQualType> Result = lookupType(V);
  if (Result.hasValue())
    return Result.getValue();

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

        // Create the struct
        std::string TypeName = getUniqueTypeNameForDecl(nullptr);
        clang::IdentifierInfo &TypeId = ASTCtx.Idents.get(TypeName);
        auto *Struct = clang::RecordDecl::Create(ASTCtx,
                                                 clang::TTK_Struct,
                                                 &DeclCtx,
                                                 clang::SourceLocation{},
                                                 clang::SourceLocation{},
                                                 &TypeId,
                                                 nullptr);

        // We have to insert it in the mapping before looking at its child,
        // otherwise we risk to generate differenty types with duplicate names.
        Result = Struct;
        insertTypeMapping(V, Result.getValue());

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
