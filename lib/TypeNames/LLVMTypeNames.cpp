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
#include "revng/PTML/CBuilder.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/Identifier.h"
#include "revng/TypeNames/LLVMTypeNames.h"
#include "revng/TypeNames/PTMLCTypeBuilder.h"

using llvm::Twine;
using CBuilder = ptml::CBuilder;
using CTypeBuilder = ptml::CTypeBuilder;

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

std::string getScalarTypeTag(const llvm::Type *LLVMType,
                             const CTypeBuilder &B) {
  revng_assert(isScalarCType(LLVMType));
  switch (LLVMType->getTypeID()) {
  case llvm::Type::HalfTyID:
  case llvm::Type::BFloatTyID:
    return B.getPrimitiveTag(model::PrimitiveKind::Float, 2);

  case llvm::Type::FloatTyID:
    return B.getPrimitiveTag(model::PrimitiveKind::Float, 4);

  case llvm::Type::DoubleTyID:
    return B.getPrimitiveTag(model::PrimitiveKind::Float, 8);

  case llvm::Type::X86_FP80TyID:
    // TODO: 80-bit float have 96 bit storage, how should we call them?
    return B.getPrimitiveTag(model::PrimitiveKind::Float, 12);

  case llvm::Type::FP128TyID:
  case llvm::Type::PPC_FP128TyID:
    return B.getPrimitiveTag(model::PrimitiveKind::Float, 16);

  case llvm::Type::VoidTyID:
    return B.getVoidTag();

  case llvm::Type::IntegerTyID: {
    auto *IntType = cast<llvm::IntegerType>(LLVMType);
    switch (IntType->getIntegerBitWidth()) {
    case 1:
      // TODO: unify this with the primitive type system.
      return B.tokenTag("bool", ptml::c::tokens::Type).toString();
    case 8:
      return B.getPrimitiveTag(model::PrimitiveKind::Unsigned, 1);
    case 16:
      return B.getPrimitiveTag(model::PrimitiveKind::Unsigned, 2);
    case 32:
      return B.getPrimitiveTag(model::PrimitiveKind::Unsigned, 4);
    case 64:
      return B.getPrimitiveTag(model::PrimitiveKind::Unsigned, 8);
    case 128:
      return B.getPrimitiveTag(model::PrimitiveKind::Unsigned, 16);
    default:
      revng_abort("Found an LLVM integer with an unsupported size.");
    }
  } break;

  case llvm::Type::PointerTyID: {
    if (uint64_t Size = B.Configuration.ExplicitTargetPointerSize) {
      return "pointer" + std::to_string(Size * 8) + "_t(" + B.getVoidTag()
             + ")";
    } else {
      return B.getVoidTag() + " "
             + B.getOperator(CBuilder::Operator::PointerDereference);
    }
  }

  default:
    revng_abort("Cannot convert this type directly to a C type.");
  }
}

static std::string getReturnedStructName(llvm::StringRef FunctionName,
                                         const CTypeBuilder &B) {
  return B.NameBuilder.Configuration.artificialReturnValuePrefix().str()
         + sanitizeIdentifier(FunctionName);
}

template<bool IsDefinition>
static std::string getReturnStructTypeTagImpl(llvm::StringRef FunctionName,
                                              const ptml::CTypeBuilder &B) {
  return B.getHelperStructTag<IsDefinition>(getReturnedStructName(FunctionName,
                                                                  B));
}

std::string getReturnStructTypeReferenceTag(llvm::StringRef FunctionName,
                                            const ptml::CTypeBuilder &B) {
  return getReturnStructTypeTagImpl<false>(FunctionName, B);
}

template<bool IsDefinition>
static std::string
getReturnTypeTagImpl(const llvm::Function *F, const CTypeBuilder &B) {
  auto *RetType = F->getReturnType();
  // Isolated functions' return types must be converted using model types
  revng_assert(not FunctionTags::Isolated.isTagOf(F));

  if (RetType->isAggregateType())
    return getReturnStructTypeTagImpl<IsDefinition>(F->getName(), B);
  else
    return getScalarTypeTag(RetType, B);
}

std::string getReturnTypeDefinitionTag(const llvm::Function *F,
                                       const CTypeBuilder &B) {
  return getReturnTypeTagImpl<true>(F, B);
}

std::string getReturnTypeReferenceTag(const llvm::Function *F,
                                      const CTypeBuilder &B) {
  return getReturnTypeTagImpl<false>(F, B);
}

std::string getReturnStructFieldTypeReferenceTag(const llvm::Function *F,
                                                 size_t Index,
                                                 const CTypeBuilder &B) {
  revng_assert(not FunctionTags::Isolated.isTagOf(F));
  auto *StructTy = llvm::cast<llvm::StructType>(F->getReturnType());
  return getScalarTypeTag(StructTy->getTypeAtIndex(Index), B);
}

template<bool IsDefinition>
static std::string getReturnStructFieldTagImpl(llvm::StringRef FunctionName,
                                               size_t Index,
                                               const CTypeBuilder &B) {
  std::string StructName = getReturnedStructName(FunctionName, B);
  std::string FieldName = (Twine(StructFieldPrefix) + Twine(Index)).str();
  revng_assert(not StructName.empty() and not FieldName.empty());
  return B.getHelperStructFieldTag<IsDefinition>(StructName, FieldName);
}

std::string getReturnStructFieldDefinitionTag(const llvm::Function *F,
                                              size_t Index,
                                              const CTypeBuilder &B) {
  revng_assert(not FunctionTags::Isolated.isTagOf(F));
  revng_assert(F->getReturnType()->isStructTy());
  return getReturnStructFieldTagImpl<true>(F->getName(), Index, B);
}

std::string getReturnStructFieldReferenceTag(llvm::StringRef FunctionName,
                                             size_t Index,
                                             const CTypeBuilder &B) {
  return getReturnStructFieldTagImpl<false>(FunctionName, Index, B);
}

std::string getReturnStructFieldReferenceTag(const llvm::Function *F,
                                             size_t Index,
                                             const CTypeBuilder &B) {
  revng_assert(not FunctionTags::Isolated.isTagOf(F));
  revng_assert(F->getReturnType()->isStructTy());
  return getReturnStructFieldReferenceTag(F->getName(), Index, B);
}

std::string getHelperFunctionDefinitionTag(const llvm::Function *F,
                                           const CTypeBuilder &B) {
  revng_assert(not FunctionTags::Isolated.isTagOf(F));
  return B.getHelperFunctionTag<true>(F->getName());
}

std::string getHelperFunctionReferenceTag(llvm::StringRef FunctionName,
                                          const CTypeBuilder &B) {
  return B.getHelperFunctionTag<false>(FunctionName);
}

std::string getHelperFunctionReferenceTag(const llvm::Function *F,
                                          const CTypeBuilder &B) {
  revng_assert(not FunctionTags::Isolated.isTagOf(F));
  return getHelperFunctionReferenceTag(F->getName(), B);
}
