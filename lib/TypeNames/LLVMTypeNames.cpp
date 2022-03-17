//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"

#include "revng/Model/Identifier.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/Support/FunctionTags.h"
#include "revng-c/TypeNames/LLVMTypeNames.h"

using llvm::Twine;
using namespace ArtificialTypes;

TypeString
getScalarCType(const llvm::Type *LLVMType, llvm::StringRef BaseType) {
  switch (LLVMType->getTypeID()) {
  case llvm::Type::HalfTyID:
  case llvm::Type::BFloatTyID:
    return TypeString("float16_t");
    break;
  case llvm::Type::FloatTyID:
    return TypeString("float32_t");
    break;
  case llvm::Type::DoubleTyID:
    return TypeString("float64_t");
    break;
  case llvm::Type::X86_FP80TyID:
    // TODO: 80-bit float have 96 bit storage, how should we call them?
    return TypeString("float96_t");
    break;
  case llvm::Type::FP128TyID:
  case llvm::Type::PPC_FP128TyID:
    return TypeString("float128_t");
    break;

  case llvm::Type::VoidTyID: {
    return TypeString("void");
    break;

  case llvm::Type::IntegerTyID: {
    auto *IntType = cast<llvm::IntegerType>(LLVMType);
    switch (IntType->getIntegerBitWidth()) {
    case 1:
      return TypeString("bool");
    case 8:
      return TypeString("uint8_t");
    case 16:
      return TypeString("uint16_t");
      break;
    case 32:
      return TypeString("uint32_t");
      break;
    case 64:
      return TypeString("uint64_t");
      break;
    case 128:
      return TypeString("uint128_t");
      break;
    default:
      revng_abort("Found an LLVM integer with a size that is not a power of "
                  "two");
    }
  } break;

  case llvm::Type::PointerTyID: {
    if (BaseType.empty())
      return TypeString("void *");
    return TypeString((BaseType + " *").str());
  } break;

  default:
    revng_abort("Cannot convert this type directly to a C type.");
  }
  }
}

TypeString getReturnType(const llvm::Function *Func) {
  auto *RetType = Func->getReturnType();
  // Isolated functions' return types must be converted using model types
  revng_assert(not FunctionTags::Isolated.isTagOf(Func));

  if (RetType->isStructTy()) {
    const auto &FuncName = model::Identifier::fromString(Func->getName());
    return TypeString((StructWrapperPrefix + FuncName).str());
  }

  return getScalarCType(RetType);
}

FieldInfo getFieldName(const llvm::StructType *StructTy, size_t Index) {
  FieldInfo FieldInfo;
  FieldInfo.FieldName = TypeString((StructFieldPrefix + Twine(Index)).str());
  FieldInfo.FieldTypeName = getScalarCType(StructTy->getTypeAtIndex(Index));
  return FieldInfo;
}
