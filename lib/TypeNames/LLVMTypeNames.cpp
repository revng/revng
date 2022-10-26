//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"

#include "revng/Model/Binary.h"
#include "revng/Model/Identifier.h"
#include "revng/Pipeline/Location.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/Pipes/Ranks.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/PTMLC.h"
#include "revng-c/TypeNames/LLVMTypeNames.h"

using llvm::Twine;
using tokenDefinition::types::TypeString;
using namespace ArtificialTypes;

namespace tokens = ptml::c::tokens;
namespace ranks = revng::ranks;

std::string getScalarCType(const llvm::Type *LLVMType) {
  switch (LLVMType->getTypeID()) {
  case llvm::Type::HalfTyID:
  case llvm::Type::BFloatTyID:
    return ptml::tokenTag("float16_t", tokens::Type).serialize();
    break;
  case llvm::Type::FloatTyID:
    return ptml::tokenTag("float32_t", tokens::Type).serialize();
    break;
  case llvm::Type::DoubleTyID:
    return ptml::tokenTag("float64_t", tokens::Type).serialize();
    break;
  case llvm::Type::X86_FP80TyID:
    // TODO: 80-bit float have 96 bit storage, how should we call them?
    return ptml::tokenTag("float96_t", tokens::Type).serialize();
    break;
  case llvm::Type::FP128TyID:
  case llvm::Type::PPC_FP128TyID:
    return ptml::tokenTag("float128_t", tokens::Type).serialize();
    break;

  case llvm::Type::VoidTyID: {
    return ptml::tokenTag("void", tokens::Type).serialize();
    break;

  case llvm::Type::IntegerTyID: {
    auto *IntType = cast<llvm::IntegerType>(LLVMType);
    switch (IntType->getIntegerBitWidth()) {
    case 1:
      return ptml::tokenTag("bool", tokens::Type).serialize();
    case 8:
      return ptml::tokenTag("uint8_t", tokens::Type).serialize();
    case 16:
      return ptml::tokenTag("uint16_t", tokens::Type).serialize();
      break;
    case 32:
      return ptml::tokenTag("uint32_t", tokens::Type).serialize();
      break;
    case 64:
      return ptml::tokenTag("uint64_t", tokens::Type).serialize();
      break;
    case 128:
      return ptml::tokenTag("uint128_t", tokens::Type).serialize();
      break;
    default:
      revng_abort("Found an LLVM integer with a size that is not a power of "
                  "two");
    }
  } break;

  case llvm::Type::PointerTyID: {
    return ptml::tokenTag("void", tokens::Type) + " "
           + operators::PointerDereference;
  } break;

  default:
    revng_abort("Cannot convert this type directly to a C type.");
  }
  }
}

tokenDefinition::types::TypeString
getReturnTypeStructName(const llvm::Function &F) {
  revng_assert(F.getReturnType()->isStructTy());
  TypeString Result = llvm::StringRef(StructWrapperPrefix);
  Result.append(model::Identifier::fromString(F.getName()));
  return Result;
}

static std::string serializeHelperStructLocation(llvm::StringRef Name) {
  return pipeline::serializedLocation(revng::ranks::HelperStruct, Name.str());
}

template<bool IsDefinition>
static std::string getReturnType(const llvm::Function *Func) {
  auto *RetType = Func->getReturnType();
  // Isolated functions' return types must be converted using model types
  revng_assert(not FunctionTags::Isolated.isTagOf(Func));

  if (RetType->isStructTy()) {
    auto StructName = getReturnTypeStructName(*Func);
    return ptml::tokenTag(StructName, tokens::Type)
      .addAttribute(ptml::locationAttribute(IsDefinition),
                    serializeHelperStructLocation(StructName))
      .serialize();
  } else {
    return getScalarCType(RetType);
  }
}

std::string getReturnTypeDefinition(const llvm::Function *Func) {
  return getReturnType<false>(Func);
}

std::string getReturnTypeReference(const llvm::Function *Func) {
  return getReturnType<true>(Func);
}

FieldInfo getFieldInfo(const llvm::StructType *StructTy, size_t Index) {
  FieldInfo FieldInfo;
  FieldInfo.FieldName = TypeString((StructFieldPrefix + Twine(Index)).str());
  std::string CTypeName = getScalarCType(StructTy->getTypeAtIndex(Index));
  FieldInfo.FieldTypeName.assign(CTypeName);
  return FieldInfo;
}
