#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include <string>

#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"

#include "revng-c/Support/TokenDefinitions.h"

/// Returns true if the LLVMTYpe is a scalar type whose name can be emitted in C
extern bool isScalarCType(const llvm::Type *LLVMType);

/// Get the C name of an LLVM Scalar type, in PTML.
extern std::string getScalarCType(const llvm::Type *LLVMType);

/// Get the PTML definition of the C name of the type returned by F.
extern std::string getReturnTypeLocationDefinition(const llvm::Function *F);

/// Get the PTML reference to the C name of the type returned by F.
extern std::string getReturnTypeLocationReference(const llvm::Function *F);

/// Get PTML tag of the C name of the type of Index-th fields of the struct type
/// returned by F.
extern std::string
getReturnStructFieldType(const llvm::Function *F, size_t Index);

/// Get the PTML definition of the C name of the Index-th field of the struct
/// returned by F.
extern std::string
getReturnStructFieldLocationDefinition(const llvm::Function *F, size_t Index);

/// Get the PTML reference to the C name of the Index-th field of the struct
/// returned by F.
extern std::string
getReturnStructFieldLocationReference(const llvm::Function *F, size_t Index);

/// Get the PTML definition of the C name of the helper function F.
extern std::string getHelperFunctionLocationDefinition(const llvm::Function *F);

/// Get the PTML reference to the C name of the helper function F.
extern std::string getHelperFunctionLocationReference(const llvm::Function *F);
