#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

namespace llvm {
class Function;
class LLVMContext;
class Module;
class Type;
class ExtractValueInst;
} // end namespace llvm

#include "revng/Support/FunctionTags.h"
#include "revng/Support/OpaqueFunctionsPool.h"

namespace FunctionTags {
extern Tag AllocatesLocalVariable;
extern Tag MallocLike;
extern Tag IsRef;
extern Tag AddressOf;
extern Tag ModelGEP;
extern Tag AssignmentMarker;
extern Tag OpaqueExtractValue;

inline Tag LiftingArtifactsRemoved("LiftingArtifactsRemoved", Isolated);

inline Tag
  StackPointerPromoted("StackPointerPromoted", LiftingArtifactsRemoved);

inline Tag
  StackAccessesSegregated("StackAccessesSegregated", StackPointerPromoted);

inline Tag DecompiledToYAML("DecompiledToYAML", StackPointerPromoted);
} // namespace FunctionTags

/// Returns the type of an AddressOf function with return type T, in context C.
/// This is intended to be used to get functions from the AddressOfPool created
/// with createAddresOfPool()
llvm::FunctionType *
getAddressOfFunctionType(llvm::LLVMContext &C, llvm::Type *T);

/// Initializes a pool of AddressOf functions, initializing it its internal
/// Module
void initAddressOfPool(OpaqueFunctionsPool<llvm::Type *> &Pool);

llvm::Function *
getModelGEP(llvm::Module &M, llvm::Type *RetTy, llvm::Type *BaseAddressTy);

llvm::Function *getAssignmentMarker(llvm::Module &M, llvm::Type *T);

/// Derive the function type of the corresponding OpaqueExtractValue() function
/// from an ExtractValue instruction. OpaqueExtractValue() basically wraps an
/// ExtractValue to prevent it from being optimized out, so the return type and
/// arguments are the same as the instruction being wrapped.
llvm::FunctionType *
getOpaqueEVFunctionType(llvm::LLVMContext &C, llvm::ExtractValueInst *Extract);

/// Key to uniquely identify an OpaqueExtractvalue in the Pool
struct TypePair {
  /// Type extracted
  llvm::Type *RetType;
  /// Aggregate type from which we are extracting a value
  llvm::Type *StructType;

  bool operator<(const TypePair &Rhs) const {
    return RetType < Rhs.RetType
           or (RetType == Rhs.RetType and StructType < Rhs.StructType);
  }
};

// Initializes a pool of OpaqueExtractValue instructions, so that a new one can
// be created on-demand.
void initOpaqueEVPool(OpaqueFunctionsPool<TypePair> &Pool, llvm::Module *M);
