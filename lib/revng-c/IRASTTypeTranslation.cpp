#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <clang/AST/ASTContext.h>

#include <revng/Support/Assert.h>

#include "IRASTTypeTranslation.h"

namespace IRASTTypeTranslation {

clang::QualType getQualType(const llvm::Type *Ty, clang::ASTContext &C) {
  clang::QualType Result;
  switch (Ty->getTypeID()) {

  case llvm::Type::TypeID::IntegerTyID: {
    auto *LLVMIntType = llvm::cast<llvm::IntegerType>(Ty);
    unsigned BitWidth = LLVMIntType->getBitWidth();
    revng_assert(BitWidth == 1 or BitWidth == 8 or BitWidth == 16
                 or BitWidth == 32 or BitWidth == 64 or BitWidth == 128);
    if (BitWidth == 1)
      BitWidth = 8;
    Result = C.getIntTypeForBitwidth(BitWidth, /* Signed */ false);
  } break;

  case llvm::Type::TypeID::PointerTyID: {
    const llvm::PointerType *PtrTy = llvm::cast<llvm::PointerType>(Ty);
    const llvm::Type *PointeeTy = PtrTy->getElementType();
    clang::QualType PointeeType = getQualType(PointeeTy, C);
    Result = C.getPointerType(PointeeType);
  } break;

  case llvm::Type::TypeID::VoidTyID:
  case llvm::Type::TypeID::MetadataTyID: {
    Result = C.VoidTy;
  } break;

  case llvm::Type::TypeID::ArrayTyID:
  case llvm::Type::TypeID::StructTyID:
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
    revng_abort();
  }
  revng_assert(not Result.isNull());
  return Result;
}

clang::QualType
getQualType(const llvm::GlobalVariable *G, clang::ASTContext &C) {
  llvm::PointerType *GlobalPtrTy = llvm::cast<llvm::PointerType>(G->getType());
  llvm::Type *Ty = GlobalPtrTy->getElementType();
  return getQualType(Ty, C);
}

clang::QualType getQualType(const llvm::Value *I, clang::ASTContext &C) {
  llvm::Type *Ty = I->getType();
  return getQualType(Ty, C);
}

} // end namespace IRASTTypeTranslation
