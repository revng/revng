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

constexpr const char *const StructWrapperPrefix = "artificial_struct_returned_"
                                                  "by_";
constexpr const char *const StructFieldPrefix = "field_";

namespace tokens = ptml::c::tokens;
namespace tags = ptml::tags;
namespace attributes = ptml::attributes;

bool isScalarCType(const llvm::Type *LLVMType) {
  switch (LLVMType->getTypeID()) {
  case llvm::Type::HalfTyID:
  case llvm::Type::BFloatTyID:
  case llvm::Type::FloatTyID:
  case llvm::Type::DoubleTyID:
  case llvm::Type::X86_FP80TyID:
  case llvm::Type::FP128TyID:
  case llvm::Type::PPC_FP128TyID:
  case llvm::Type::VoidTyID:
  case llvm::Type::PointerTyID:
    return true;

  case llvm::Type::IntegerTyID: {
    auto *IntType = cast<llvm::IntegerType>(LLVMType);
    switch (IntType->getIntegerBitWidth()) {
    case 1:
    case 8:
    case 16:
    case 32:
    case 64:
    case 128:
      return true;
    default:
      return false;
    }
  } break;

  default:
    return false;
  }
  return false;
}

std::string getScalarCType(const llvm::Type *LLVMType) {
  revng_assert(isScalarCType(LLVMType));
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

static std::string getHelperFunctionIdentifier(const llvm::Function *F) {
  revng_assert(not FunctionTags::Isolated.isTagOf(F));
  return model::Identifier::fromString(F->getName()).str().str();
}

static std::string getReturnedStructIdentifier(const llvm::Function *F) {
  revng_assert(not FunctionTags::Isolated.isTagOf(F));
  revng_assert(llvm::isa<llvm::StructType>(F->getReturnType()));
  return (Twine(StructWrapperPrefix) + Twine(getHelperFunctionIdentifier(F)))
    .str();
}

static std::string serializeHelperStructLocation(const std::string &Name) {
  return pipeline::serializedLocation(revng::ranks::HelperStructType, Name);
}

template<bool IsDefinition>
static std::string getReturnTypeLocation(const llvm::Function *F) {
  auto *RetType = F->getReturnType();
  // Isolated functions' return types must be converted using model types
  revng_assert(not FunctionTags::Isolated.isTagOf(F));

  if (RetType->isAggregateType()) {
    std::string StructName = getReturnedStructIdentifier(F);
    return ptml::tokenTag(StructName, tokens::Type)
      .addAttribute(ptml::locationAttribute(IsDefinition),
                    serializeHelperStructLocation(StructName))
      .serialize();
  } else {
    return getScalarCType(RetType);
  }
}

std::string getReturnTypeLocationDefinition(const llvm::Function *F) {
  return getReturnTypeLocation<false>(F);
}

std::string getReturnTypeLocationReference(const llvm::Function *F) {
  return getReturnTypeLocation<true>(F);
}

std::string getReturnStructFieldType(const llvm::Function *F, size_t Index) {
  revng_assert(not FunctionTags::Isolated.isTagOf(F));
  auto *StructTy = llvm::cast<llvm::StructType>(F->getReturnType());
  return getScalarCType(StructTy->getTypeAtIndex(Index));
}

static std::string
serializeHelperStructFieldLocation(const std::string &StructName,
                                   const std::string &FieldName) {
  revng_assert(not StructName.empty() and not FieldName.empty());
  return pipeline::serializedLocation(revng::ranks::HelperStructField,
                                      StructName,
                                      FieldName);
}

template<bool IsDefinition>
static std::string
getReturnStructFieldLocation(const llvm::Function *F, size_t Index) {
  revng_assert(not FunctionTags::Isolated.isTagOf(F));
  revng_assert(F->getReturnType()->isStructTy());

  std::string StructName = getReturnedStructIdentifier(F);
  std::string FieldName = (Twine(StructFieldPrefix) + Twine(Index)).str();
  return ptml::Tag(tags::Span, FieldName)
    .addAttribute(attributes::Token, tokens::Field)
    .addAttribute(ptml::locationAttribute(IsDefinition),
                  serializeHelperStructFieldLocation(StructName, FieldName))
    .serialize();
}

std::string
getReturnStructFieldLocationDefinition(const llvm::Function *F, size_t Index) {
  return getReturnStructFieldLocation<true>(F, Index);
}

std::string
getReturnStructFieldLocationReference(const llvm::Function *F, size_t Index) {
  return getReturnStructFieldLocation<false>(F, Index);
}

static std::string serializeHelperFunctionLocation(const llvm::Function *F) {
  return pipeline::serializedLocation(revng::ranks::HelperFunction,
                                      F->getName().str());
}

template<bool IsDefinition>
static std::string getHelperFunctionLocation(const llvm::Function *F) {
  return ptml::tokenTag(getHelperFunctionIdentifier(F), tokens::Function)
    .addAttribute(ptml::locationAttribute(IsDefinition),
                  serializeHelperFunctionLocation(F))
    .serialize();
}

std::string getHelperFunctionLocationDefinition(const llvm::Function *F) {
  return getHelperFunctionLocation<true>(F);
}

std::string getHelperFunctionLocationReference(const llvm::Function *F) {
  return getHelperFunctionLocation<false>(F);
}
