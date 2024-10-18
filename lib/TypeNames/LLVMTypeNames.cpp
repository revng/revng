//
// This file is distributed under the MIT License. See LICENSE.md for details.
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
#include "revng-c/Support/PTMLC.h"
#include "revng-c/TypeNames/LLVMTypeNames.h"
#include "revng-c/TypeNames/PTMLCTypeBuilder.h"

using llvm::Twine;
using CBuilder = ptml::CBuilder;
using CTypeBuilder = ptml::CTypeBuilder;

constexpr const char *const StructWrapperPrefix = "_artificial_struct_returned_"
                                                  "by_";
constexpr const char *const StructFieldPrefix = "field_";

namespace tokens = ptml::c::tokens;
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
  }

  default:
    return false;
  }
  return false;
}

std::string getScalarCType(const llvm::Type *LLVMType, const CBuilder &B) {
  revng_assert(isScalarCType(LLVMType));
  using Operator = CBuilder::Operator;
  switch (LLVMType->getTypeID()) {
  case llvm::Type::HalfTyID:
  case llvm::Type::BFloatTyID:
    return B.tokenTag("float16_t", ptml::c::tokens::Type).toString();

  case llvm::Type::FloatTyID:
    return B.tokenTag("float32_t", ptml::c::tokens::Type).toString();

  case llvm::Type::DoubleTyID:
    return B.tokenTag("float64_t", ptml::c::tokens::Type).toString();

  case llvm::Type::X86_FP80TyID:
    // TODO: 80-bit float have 96 bit storage, how should we call them?
    return B.tokenTag("float96_t", ptml::c::tokens::Type).toString();

  case llvm::Type::FP128TyID:
  case llvm::Type::PPC_FP128TyID:
    return B.tokenTag("float128_t", ptml::c::tokens::Type).toString();

  case llvm::Type::VoidTyID: {
    return B.tokenTag("void", ptml::c::tokens::Type).toString();

  case llvm::Type::IntegerTyID: {
    auto *IntType = cast<llvm::IntegerType>(LLVMType);
    switch (IntType->getIntegerBitWidth()) {
    case 1:
      return B.tokenTag("bool", ptml::c::tokens::Type).toString();
    case 8:
      return B.tokenTag("uint8_t", ptml::c::tokens::Type).toString();
    case 16:
      return B.tokenTag("uint16_t", ptml::c::tokens::Type).toString();
    case 32:
      return B.tokenTag("uint32_t", ptml::c::tokens::Type).toString();
    case 64:
      return B.tokenTag("uint64_t", ptml::c::tokens::Type).toString();
    case 128:
      return B.tokenTag("uint128_t", ptml::c::tokens::Type).toString();
    default:
      revng_abort("Found an LLVM integer with a size that is not a power of "
                  "two");
    }
  } break;

  case llvm::Type::PointerTyID: {
    return B.tokenTag("void", ptml::c::tokens::Type) + " "
           + B.getOperator(Operator::PointerDereference);
  }

  default:
    revng_abort("Cannot convert this type directly to a C type.");
  }
  }
}

static std::string getHelperFunctionIdentifier(const llvm::Function *F) {
  revng_assert(not FunctionTags::Isolated.isTagOf(F));
  return (Twine("_") + model::Identifier::sanitize(F->getName())).str();
}

static std::string getReturnedStructIdentifier(const llvm::Function *F) {
  revng_assert(not FunctionTags::Isolated.isTagOf(F));
  revng_assert(llvm::isa<llvm::StructType>(F->getReturnType()));
  return (Twine(StructWrapperPrefix) + Twine(getHelperFunctionIdentifier(F)))
    .str();
}

static std::string serializeHelperStructLocation(const std::string &Name) {
  return pipeline::locationString(revng::ranks::HelperStructType, Name);
}

template<bool IsDefinition>
static std::string
getReturnTypeLocation(const llvm::Function *F, const CTypeBuilder &B) {
  auto *RetType = F->getReturnType();
  // Isolated functions' return types must be converted using model types
  revng_assert(not FunctionTags::Isolated.isTagOf(F));

  if (RetType->isAggregateType()) {
    std::string StructName = getReturnedStructIdentifier(F);
    return B.tokenTag(StructName, ptml::c::tokens::Type)
      .addAttribute(B.getLocationAttribute(IsDefinition),
                    serializeHelperStructLocation(StructName))
      .toString();
  } else {
    return getScalarCType(RetType, B);
  }
}

std::string getReturnTypeLocationDefinition(const llvm::Function *F,
                                            const CTypeBuilder &B) {
  return getReturnTypeLocation<false>(F, B);
}

std::string getReturnTypeLocationReference(const llvm::Function *F,
                                           const CTypeBuilder &B) {
  return getReturnTypeLocation<true>(F, B);
}

std::string getReturnStructFieldType(const llvm::Function *F,
                                     size_t Index,
                                     const CBuilder &B) {
  revng_assert(not FunctionTags::Isolated.isTagOf(F));
  auto *StructTy = llvm::cast<llvm::StructType>(F->getReturnType());
  return getScalarCType(StructTy->getTypeAtIndex(Index), B);
}

static std::string
serializeHelperStructFieldLocation(const std::string &StructName,
                                   const std::string &FieldName) {
  revng_assert(not StructName.empty() and not FieldName.empty());
  return pipeline::locationString(revng::ranks::HelperStructField,
                                  StructName,
                                  FieldName);
}

template<bool IsDefinition>
static std::string getReturnStructFieldLocation(const llvm::Function *F,
                                                size_t Index,
                                                const CTypeBuilder &B) {
  revng_assert(not FunctionTags::Isolated.isTagOf(F));
  revng_assert(F->getReturnType()->isStructTy());

  std::string StructName = getReturnedStructIdentifier(F);
  std::string FieldName = (Twine(StructFieldPrefix) + Twine(Index)).str();
  return B.getTag(ptml::tags::Span, FieldName)
    .addAttribute(attributes::Token, tokens::Field)
    .addAttribute(B.getLocationAttribute(IsDefinition),
                  serializeHelperStructFieldLocation(StructName, FieldName))
    .toString();
}

std::string getReturnStructFieldLocationDefinition(const llvm::Function *F,
                                                   size_t Index,
                                                   const CTypeBuilder &B) {
  return getReturnStructFieldLocation<true>(F, Index, B);
}

std::string getReturnStructFieldLocationReference(const llvm::Function *F,
                                                  size_t Index,
                                                  const CTypeBuilder &B) {
  return getReturnStructFieldLocation<false>(F, Index, B);
}

static std::string serializeHelperFunctionLocation(const llvm::Function *F) {
  return pipeline::locationString(revng::ranks::HelperFunction,
                                  F->getName().str());
}

template<bool IsDefinition>
static std::string
getHelperFunctionLocation(const llvm::Function *F, const CTypeBuilder &B) {
  return B.tokenTag(getHelperFunctionIdentifier(F), ptml::c::tokens::Function)
    .addAttribute(B.getLocationAttribute(IsDefinition),
                  serializeHelperFunctionLocation(F))
    .toString();
}

std::string getHelperFunctionLocationDefinition(const llvm::Function *F,
                                                const CTypeBuilder &B) {
  return getHelperFunctionLocation<true>(F, B);
}

std::string getHelperFunctionLocationReference(const llvm::Function *F,
                                               const CTypeBuilder &B) {
  return getHelperFunctionLocation<false>(F, B);
}
