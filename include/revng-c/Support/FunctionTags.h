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
extern Tag ModelCast;
extern Tag AssignmentMarker;
extern Tag OpaqueExtractValue;
extern Tag Parentheses;
extern Tag ReadsMemory;
extern Tag WritesMemory;

inline Tag LiftingArtifactsRemoved("LiftingArtifactsRemoved", Isolated);

inline Tag
  StackPointerPromoted("StackPointerPromoted", LiftingArtifactsRemoved);

inline Tag
  StackAccessesSegregated("StackAccessesSegregated", StackPointerPromoted);

inline Tag DecompiledToYAML("DecompiledToYAML", StackPointerPromoted);
} // namespace FunctionTags

inline const llvm::CallInst *
isCallToTagged(const llvm::Value *V, FunctionTags::Tag &T) {
  if (auto *Call = llvm::dyn_cast<llvm::CallInst>(V))
    if (auto *CalledFunc = Call->getCalledFunction())
      if (T.isTagOf(CalledFunc))
        return Call;

  return nullptr;
}

inline llvm::CallInst *isCallToTagged(llvm::Value *V, FunctionTags::Tag &T) {
  if (auto *Call = llvm::dyn_cast<llvm::CallInst>(V))
    if (auto *CalledFunc = Call->getCalledFunction())
      if (T.isTagOf(CalledFunc))
        return Call;

  return nullptr;
}

/// This struct can be used as a key of an OpaqueFunctionsPool where both
/// the return type and one of the arguments are needed to identify a function
/// in the pool.
struct TypePair {
  llvm::Type *RetType;
  llvm::Type *ArgType;

  bool operator<(const TypePair &Rhs) const {
    return RetType < Rhs.RetType
           or (RetType == Rhs.RetType and ArgType < Rhs.ArgType);
  }
};

/// AddressOf functions are used to transform a reference into a pointer.
///
/// \param RetType The LLVM type returned by the Addressof call
/// \param BaseType The LLVM type of the second argument (the reference that
/// we want to transform into a pointer).
llvm::FunctionType *getAddressOfType(llvm::Type *RetType, llvm::Type *BaseType);

/// Initializes a pool of AddressOf functions, initializing it its internal
/// Module
void initAddressOfPool(OpaqueFunctionsPool<TypePair> &Pool, llvm::Module *M);

/// Initializes a pool of Parentheses functions
void initParenthesesPool(OpaqueFunctionsPool<llvm::Type *> &Pool);

/// ModelGEP functions are used to replace pointer arithmetics with a navigation
/// of the Model.
///
/// \param RetType ModelGEP should return an integer of the size of the gepped
/// element
/// \param BaseType The LLVM type of the second argument (the base pointer)
llvm::Function *
getModelGEP(llvm::Module &M, llvm::Type *RetType, llvm::Type *BaseType);

/// Initializes a pool of ModelCast functions
void initModelCastPool(OpaqueFunctionsPool<llvm::Type *> &Pool);

llvm::Function *getAssignmentMarker(llvm::Module &M, llvm::Type *T);

/// Derive the function type of the corresponding OpaqueExtractValue() function
/// from an ExtractValue instruction. OpaqueExtractValues wrap an
/// ExtractValue to prevent it from being optimized out, so the return type and
/// arguments are the same as the instruction being wrapped.
llvm::FunctionType *getOpaqueEVFunctionType(llvm::ExtractValueInst *Extract);

// Initializes a pool of OpaqueExtractValue instructions, so that a new one can
// be created on-demand.
void initOpaqueEVPool(OpaqueFunctionsPool<TypePair> &Pool, llvm::Module *M);
