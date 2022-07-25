#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"

#include "revng-c/Support/TokenDefinitions.h"

namespace tokenTypes = tokenDefinition::types;
namespace ArtificialTypes {
constexpr const char *const StructWrapperPrefix = "artificial_struct_";
constexpr const char *const StructFieldPrefix = "field_";
} // namespace ArtificialTypes

/// Print the C name of an LLVM Scalar type.
/// \note Pointer types use the \a BaseType provided if it's not empty,
/// otherwise they are converted as `void *`
extern tokenTypes::TypeString
getScalarCType(const llvm::Type *LLVMType, llvm::StringRef BaseType = "");

/// Get the name of the type returned by an llvm::Function.
/// \note Do not use this for isolated functions - use the Model prototype
/// instead
extern VariableTokensWithName getReturnType(const llvm::Function *Func);

struct FieldInfo {
  tokenTypes::TypeString FieldName;
  tokenTypes::TypeString FieldTypeName;
};

/// Return the name and type of the \a Index -th field of a struct type.
extern FieldInfo getFieldInfo(const llvm::StructType *StructTy, size_t Index);
