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
extern bool isScalarCType(const llvm::Type *LLVMType);

/// Get the C name of an LLVM Scalar type, in PTML.
extern std::string getScalarCType(const llvm::Type *LLVMType,
                                  const ptml::CBuilder &B);

/// Get the PTML definition of the C name of the type returned by F.
extern std::string getReturnTypeLocationDefinition(const llvm::Function *F,
                                                   const ptml::CTypeBuilder &B);

/// Get the PTML reference to the C name of the type returned by F.
extern std::string getReturnTypeLocationReference(const llvm::Function *F,
                                                  const ptml::CTypeBuilder &B);

/// Get PTML tag of the C name of the type of Index-th fields of the struct type
/// returned by F.
extern std::string getReturnStructFieldType(const llvm::Function *F,
                                            size_t Index,
                                            const ptml::CBuilder &B);

/// Get the PTML definition of the C name of the Index-th field of the struct
/// returned by F.
extern std::string
getReturnStructFieldLocationDefinition(const llvm::Function *F,
                                       size_t Index,
                                       const ptml::CTypeBuilder &B);

/// Get the PTML reference to the C name of the Index-th field of the struct
/// returned by F.
extern std::string
getReturnStructFieldLocationReference(const llvm::Function *F,
                                      size_t Index,
                                      const ptml::CTypeBuilder &B);

/// Get the PTML definition of the C name of the helper function F.
extern std::string
getHelperFunctionLocationDefinition(const llvm::Function *F,
                                    const ptml::CTypeBuilder &B);

/// Get the PTML reference to the C name of the helper function F.
extern std::string
getHelperFunctionLocationReference(const llvm::Function *F,
                                   const ptml::CTypeBuilder &B);
