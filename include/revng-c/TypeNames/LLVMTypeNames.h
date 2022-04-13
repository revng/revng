#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/SmallString.h"

namespace llvm {
class StringRef;
class StructType;
class Function;
class Type;
} // namespace llvm

namespace ArtificialTypes {
constexpr const char *const StructWrapperPrefix = "artificial_struct_";
constexpr const char *const StructFieldPrefix = "field_";
} // namespace ArtificialTypes

using TypeString = llvm::SmallString<32>;

/// Print the C name of an LLVM Scalar type.
/// \note Pointer types use the \a BaseType provided if it's not empty,
/// otherwise they are converted as `void *`
extern TypeString
getScalarCType(const llvm::Type *LLVMType, llvm::StringRef BaseType = "");

/// Get the name of the type returned by an llvm::Function.
/// \note Do not use this for isolated functions - use the Model prototype
/// instead
extern TypeString getReturnType(const llvm::Function *Func);

struct FieldInfo {
  TypeString FieldName;
  TypeString FieldTypeName;
};

/// Return the name and type of the \a Index -th field of a struct type.
extern FieldInfo getFieldName(const llvm::StructType *StructTy, size_t Index);
