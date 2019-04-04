// LLVM includes
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

// clang includes
#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>

// revng includes
#include <revng/Support/Assert.h>

// local includes
#include "IRASTTypeTranslation.h"
#include "Mangling.h"

namespace IRASTTypeTranslation {

clang::QualType getOrCreateQualType(const llvm::Type *Ty,
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
      const std::string BoolName = "bool";
      clang::IdentifierInfo &Id = ASTCtx.Idents.get(BoolName);
      clang::DeclarationName TypeName(&Id);
      for (clang::Decl *D : TUDecl->lookup(TypeName)) {
        if (auto *Typedef = llvm::dyn_cast<clang::TypedefDecl>(D)) {
          const std::string Name = Typedef->getNameAsString();
          if (Name == "bool") {
            Result = ASTCtx.getTypedefType(Typedef);
            return Result;
          }
        }
      }
      revng_abort("'bool' type not found!\n"
                  "this should not happen since we '#include <stdbool.h>'\n"
                  "please make sure you have installed the header\n");
    } break;
    case 16: {
      const std::string UInt16Name = "uint16_t";
      clang::IdentifierInfo &Id = ASTCtx.Idents.get(UInt16Name);
      clang::DeclarationName TypeName(&Id);
      for (clang::Decl *D : TUDecl->lookup(TypeName)) {
        if (auto *Typedef = llvm::dyn_cast<clang::TypedefDecl>(D)) {
          Result = ASTCtx.getTypedefType(Typedef);
          return Result;
        }
      }
      revng_abort("'uint16_t' type not found!\n"
                  "this should not happen since we '#include <stdint.h>'\n"
                  "please make sure you have installed the header\n");
    } break;
    case 32: {
      const std::string UInt32Name = "uint32_t";
      clang::IdentifierInfo &Id = ASTCtx.Idents.get(UInt32Name);
      clang::DeclarationName TypeName(&Id);
      for (clang::Decl *D : TUDecl->lookup(TypeName)) {
        if (auto *Typedef = llvm::dyn_cast<clang::TypedefDecl>(D)) {
          Result = ASTCtx.getTypedefType(Typedef);
          return Result;
        }
      }
      revng_abort("'uint32_t' type not found!\n"
                  "this should not happen since we '#include <stdint.h>'\n"
                  "please make sure you have installed the header\n");
    } break;
    case 64: {
      const std::string UInt64Name = "uint64_t";
      clang::IdentifierInfo &Id = ASTCtx.Idents.get(UInt64Name);
      clang::DeclarationName TypeName(&Id);
      for (clang::Decl *D : TUDecl->lookup(TypeName)) {
        if (auto *Typedef = llvm::dyn_cast<clang::TypedefDecl>(D)) {
          Result = ASTCtx.getTypedefType(Typedef);
          return Result;
        }
      }
      revng_abort("'uint64_t' type not found!\n"
                  "this should not happen since we '#include <stdint.h>'\n"
                  "please make sure you have installed the header\n");
    } break;
    case 8:
      // use char for strict aliasing reasons
    case 128:
      // use builtin clang types (__int128_t and __uint128_t), because C99 does
      // not require to have 128 bit integer types
      break;
    default:
      revng_abort("unexpected integer size");
    }
    Result = ASTCtx.getIntTypeForBitwidth(BitWidth, /* Signed */ false);
  } break;

  case llvm::Type::TypeID::PointerTyID: {
    const llvm::PointerType *PtrTy = llvm::cast<llvm::PointerType>(Ty);
    const llvm::Type *PointeeTy = PtrTy->getElementType();
    clang::QualType PointeeType = getOrCreateQualType(PointeeTy,
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
    if (StructTy->hasName())
      TypeName = makeCIdentifier(StructTy->getName());
    else
      TypeName = "type_" + std::to_string(TypeDecls.size());
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
      clang::QualType QFieldTy = getOrCreateQualType(FieldTy,
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
  return getOrCreateQualType(Ty, ASTCtx, DeclCtx, TypeDecls, FieldDecls);
}

clang::QualType getOrCreateQualType(const llvm::Value *I,
                                    clang::ASTContext &ASTCtx,
                                    clang::DeclContext &DeclCtx,
                                    TypeDeclMap &TypeDecls,
                                    FieldDeclMap &FieldDecls) {
  llvm::Type *Ty = I->getType();
  return getOrCreateQualType(Ty, ASTCtx, DeclCtx, TypeDecls, FieldDecls);
}

} // end namespace IRASTTypeTranslation
