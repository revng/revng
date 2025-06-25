#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"

namespace llvm {
class Function;
class Type;
} // namespace llvm

namespace ptml {
class CBuilder;
class ModelCBuilder;
} // namespace ptml

/// Returns true if the LLVMTYpe is a scalar type whose name can be emitted in C
bool isScalarCType(const llvm::Type *LLVMType);

/// Get the C name of an LLVM Scalar type, in PTML.
std::string getScalarTypeTag(const llvm::Type *LLVMType,
                             const ptml::ModelCBuilder &B);

/// Get the PTML reference to the C name of the type returned by the helper
/// function with the specified name.
std::string getReturnStructTypeReferenceTag(llvm::StringRef FunctionName,
                                            const ptml::ModelCBuilder &B);

/// Get the PTML definition of the C name of the type returned by F.
std::string getReturnTypeDefinitionTag(const llvm::Function *F,
                                       const ptml::ModelCBuilder &B);

/// Get the PTML reference to the C name of the type returned by F.
std::string getReturnTypeReferenceTag(const llvm::Function *F,
                                      const ptml::ModelCBuilder &B);

/// Get PTML tag of the C name of the type of Index-th fields of the struct type
/// returned by F.
std::string getReturnStructFieldTypeReferenceTag(const llvm::Function *F,
                                                 size_t Index,
                                                 const ptml::ModelCBuilder &B);

/// Get the PTML definition of the C name of the Index-th field of the struct
/// returned by F.
std::string getReturnStructFieldDefinitionTag(const llvm::Function *F,
                                              size_t Index,
                                              const ptml::ModelCBuilder &B);

/// Get the PTML reference to the C name of the Index-th field of the struct
/// returned by the helper function with the specified name.
std::string getReturnStructFieldReferenceTag(llvm::StringRef FunctionName,
                                             size_t Index,
                                             const ptml::ModelCBuilder &B);

/// Get the PTML reference to the C name of the Index-th field of the struct
/// returned by F.
std::string getReturnStructFieldReferenceTag(const llvm::Function *F,
                                             size_t Index,
                                             const ptml::ModelCBuilder &B);

/// Get the PTML definition of the C name of the helper function F.
std::string getHelperFunctionDefinitionTag(const llvm::Function *F,
                                           const ptml::ModelCBuilder &B);

/// Get the PTML reference to the C name of the helper function with the
/// specified name.
std::string getHelperFunctionReferenceTag(llvm::StringRef FunctionName,
                                          const ptml::ModelCBuilder &B);

/// Get the PTML reference to the C name of the helper function F.
std::string getHelperFunctionReferenceTag(const llvm::Function *F,
                                          const ptml::ModelCBuilder &B);
