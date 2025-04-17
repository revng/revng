#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

namespace llvm {
class Function;
class Type;
} // namespace llvm

namespace ptml {
class CBuilder;
class CTypeBuilder;
} // namespace ptml

/// Returns true if the LLVMTYpe is a scalar type whose name can be emitted in C
bool isScalarCType(const llvm::Type *LLVMType);

/// Returns true if F is a helper whose name can be emitted in C
bool isPrintableHelper(const llvm::Function &F);

/// Get the C name of an LLVM Scalar type, in PTML.
std::string getScalarCType(const llvm::Type *LLVMType, const ptml::CBuilder &B);

/// Get the PTML definition of the C name of the type returned by F.
std::string getReturnTypeLocationDefinition(const llvm::Function *F,
                                            const ptml::CTypeBuilder &B);

/// Get the PTML reference to the C name of the type returned by F.
std::string getReturnTypeLocationReference(const llvm::Function *F,
                                           const ptml::CTypeBuilder &B);

/// Get PTML tag of the C name of the type of Index-th fields of the struct type
/// returned by F.
std::string getReturnStructFieldType(const llvm::Function *F,
                                     size_t Index,
                                     const ptml::CBuilder &B);

/// Get the PTML definition of the C name of the Index-th field of the struct
/// returned by F.
std::string getReturnStructFieldLocationDefinition(const llvm::Function *F,
                                                   size_t Index,
                                                   const ptml::CTypeBuilder &B);

/// Get the PTML reference to the C name of the Index-th field of the struct
/// returned by F.
std::string getReturnStructFieldLocationReference(const llvm::Function *F,
                                                  size_t Index,
                                                  const ptml::CTypeBuilder &B);

/// Get the PTML definition of the C name of the helper function F.
std::string getHelperFunctionLocationDefinition(const llvm::Function *F,
                                                const ptml::CTypeBuilder &B);

/// Get the PTML reference to the C name of the helper function F.
std::string getHelperFunctionLocationReference(const llvm::Function *F,
                                               const ptml::CTypeBuilder &B);
