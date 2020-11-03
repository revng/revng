// LLVM includes
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

// clang includes
#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>

// revng includes
#include <revng/Support/Assert.h>

// local includes
#include "IRASTTypeTranslation.h"
#include "Mangling.h"

namespace IRASTTypeTranslation {

clang::QualType getOrCreateBoolQualType(clang::ASTContext &ASTCtx,
                                        TypeDeclMap &TypeDecls,
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
        if (Ty != nullptr) {
          auto It = TypeDecls.begin();
          bool NewInsert = false;
          std::tie(It, NewInsert) = TypeDecls.insert({ Ty, Typedef });
          revng_assert(NewInsert or It->second == Typedef);
        }
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
    if (Ty != nullptr)
      TypeDecls[Ty] = BoolTypedefDecl;
    TUDecl->addDecl(BoolTypedefDecl);
    Result = ASTCtx.getTypedefType(BoolTypedefDecl);
    return Result;
    revng_abort("'_Bool' type not found!\n"
                "This should not happen since we compile as c99\n");
  }
  return Result;
}

clang::QualType getOrCreateQualType(const llvm::Type *Ty,
                                    const llvm::Value *NamingValue,
                                    clang::ASTContext &ASTCtx,
                                    clang::DeclContext &DeclCtx,
                                    TypeDeclMap &TypeDecls,
                                    FieldDeclMap &FieldDecls) {
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
      Result = getOrCreateBoolQualType(ASTCtx, TypeDecls, Ty);
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
                                                      DeclCtx,
                                                      TypeDecls,
                                                      FieldDecls);
    Result = ASTCtx.getPointerType(PointeeType);
  } break;

  case llvm::Type::TypeID::VoidTyID:
  case llvm::Type::TypeID::MetadataTyID: {
    Result = ASTCtx.VoidTy;
  } break;

  case llvm::Type::TypeID::StructTyID: {
    auto It = TypeDecls.find(Ty);
    if (It != TypeDecls.end()) {
      Result = clang::QualType(It->second->getTypeForDecl(), 0);
      break;
    }

    const llvm::StructType *StructTy = llvm::cast<llvm::StructType>(Ty);
    std::string TypeName;
    if (StructTy->hasName()) {
      TypeName = makeCIdentifier(StructTy->getName());
    } else if (NamingValue and NamingValue->hasName()) {
      if (auto *F = llvm::dyn_cast_or_null<llvm::Function>(NamingValue))
        TypeName = makeCIdentifier(F->getName().str())
                   + std::string("_ret_type");
      else
        TypeName = makeCIdentifier(NamingValue->getName().str())
                   + std::string("_type");
    } else {
      TypeName = "type_" + std::to_string(TypeDecls.size());
    }
    clang::IdentifierInfo &TypeId = ASTCtx.Idents.get(TypeName);
    auto *Struct = clang::RecordDecl::Create(ASTCtx,
                                             clang::TTK_Struct,
                                             &DeclCtx,
                                             clang::SourceLocation{},
                                             clang::SourceLocation{},
                                             &TypeId,
                                             nullptr);
    TypeDecls[Ty] = Struct;
    unsigned N = 0;
    FieldDecls[Struct].resize(StructTy->getNumElements(), nullptr);
    Struct->startDefinition();
    for (llvm::Type *FieldTy : StructTy->elements()) {

      // HACK: Handle the type of the `@env` global variable, which we simply
      //       cast to a `void` `nullptr` type.
      if (FieldTy->getTypeID() == llvm::Type::TypeID::ArrayTyID) {
        Result = ASTCtx.VoidTy;
        break;
      }
      clang::QualType QFieldTy = getOrCreateQualType(FieldTy,
                                                     nullptr,
                                                     ASTCtx,
                                                     DeclCtx,
                                                     TypeDecls,
                                                     FieldDecls);
      clang::TypeSourceInfo *TI = ASTCtx.CreateTypeSourceInfo(QFieldTy);
      const std::string FieldName = std::string("field_") + std::to_string(N);
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
      FieldDecls[Struct][N] = Field;
      Struct->addDecl(Field);
      N++;
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

clang::QualType getOrCreateQualType(const llvm::GlobalVariable *G,
                                    clang::ASTContext &ASTCtx,
                                    clang::DeclContext &DeclCtx,
                                    TypeDeclMap &TypeDecls,
                                    FieldDeclMap &FieldDecls) {
  llvm::PointerType *GlobalPtrTy = llvm::cast<llvm::PointerType>(G->getType());
  llvm::Type *Ty = GlobalPtrTy->getElementType();
  return getOrCreateQualType(Ty, G, ASTCtx, DeclCtx, TypeDecls, FieldDecls);
}

clang::QualType getOrCreateQualType(const llvm::Value *I,
                                    clang::ASTContext &ASTCtx,
                                    clang::DeclContext &DeclCtx,
                                    TypeDeclMap &TypeDecls,
                                    FieldDeclMap &FieldDecls) {
  llvm::Type *Ty = I->getType();
  return getOrCreateQualType(Ty, I, ASTCtx, DeclCtx, TypeDecls, FieldDecls);
}

} // end namespace IRASTTypeTranslation
