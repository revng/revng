#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>
namespace llvm {
class Function;
class LLVMContext;
class Module;
class Type;
class ExtractValueInst;
} // end namespace llvm

#include "revng/Support/FunctionTags.h"
#include "revng/Support/OpaqueFunctionsPool.h"

/// AddressOf functions are used to transform a reference into a pointer.
///
/// \param RetType The LLVM type returned by the Addressof call
/// \param BaseType The LLVM type of the second argument (the reference that
/// we want to transform into a pointer).
llvm::FunctionType *getAddressOfType(llvm::Type *RetType, llvm::Type *BaseType);

/// ModelGEP functions are used to replace pointer arithmetic with a navigation
/// of the Model.
///
/// \param RetType ModelGEP should return an integer of the size of the gepped
/// element
/// \param BaseType The LLVM type of the second argument (the base pointer)
llvm::Function *
getModelGEP(llvm::Module &M, llvm::Type *RetType, llvm::Type *BaseType);

/// ModelGEP Ref is a ModelGEP where the base value is considered to be a
/// reference.
llvm::Function *
getModelGEPRef(llvm::Module &M, llvm::Type *RetType, llvm::Type *BaseType);

using ModelCastPoolKey = std::pair<llvm::Type *, llvm::Type *>;

/// Derive the function type of the corresponding OpaqueExtractValue() function
/// from an ExtractValue instruction. OpaqueExtractValues wrap an
/// ExtractValue to prevent it from being optimized out, so the return type and
/// arguments are the same as the instruction being wrapped.
llvm::FunctionType *getOpaqueEVFunctionType(llvm::ExtractValueInst *Extract);

/// LocalVariable is used to indicate the allocation of a local variable. It
/// returns a reference to the allocated variable.
llvm::FunctionType *getLocalVarType(llvm::Type *ReturnedType);

/// Assign() are meant to replace `store` instructions in which the pointer
/// operand is a reference.
llvm::FunctionType *getAssignFunctionType(llvm::Type *ValueType,
                                          llvm::Type *PtrType);

/// Copy() are meant to replace `load` instructions in which the pointer
/// operand is a reference.
llvm::FunctionType *getCopyType(llvm::Type *ReturnedType);
