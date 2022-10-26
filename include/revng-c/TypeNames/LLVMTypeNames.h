#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include <string>

#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"

#include "revng-c/Support/TokenDefinitions.h"

namespace ArtificialTypes {
constexpr const char *const StructWrapperPrefix = "artificial_struct_";
constexpr const char *const StructFieldPrefix = "field_";
} // namespace ArtificialTypes

/// Get the C name of an LLVM Scalar type, in PTML.
extern std::string getScalarCType(const llvm::Type *LLVMType);

/// Get the name of the type returned by an llvm::Function in PTML.
extern std::string getReturnTypeReference(const llvm::Function *Func);

/// Get the name of the type returned by an llvm::Function in PTML.
extern std::string getReturnTypeDefinition(const llvm::Function *Func);

struct FieldInfo {
  tokenDefinition::types::TypeString FieldName;
  tokenDefinition::types::TypeString FieldTypeName;
};

/// Return the name and type of the \a Index -th field of a struct type.
extern FieldInfo getFieldInfo(const llvm::StructType *StructTy, size_t Index);

extern tokenDefinition::types::TypeString
getReturnTypeStructName(const llvm::Function &F);
