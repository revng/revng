#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Type.h"

#include "revng/Model/Architecture.h"
#include "revng/Model/Binary.h"
#include "revng/Model/TypeDefinition.h"
#include "revng/Support/FunctionTags.h"

inline bool hasSideEffects(const llvm::Instruction &I) {
  if (llvm::isa<llvm::StoreInst>(I))
    return true;

  auto *Call = llvm::dyn_cast<llvm::CallInst>(&I);
  if (not Call)
    return false;

  auto CallMemoryEffects = Call->getMemoryEffects();
  bool MayAccessMemory = not CallMemoryEffects.doesNotAccessMemory();
  bool OnlyReadsMemory = CallMemoryEffects.onlyReadsMemory();
  return MayAccessMemory and not OnlyReadsMemory;
}

inline bool mayReadMemory(const llvm::Instruction &I) {
  if (llvm::isa<llvm::LoadInst>(I))
    return true;

  auto *Call = llvm::dyn_cast<llvm::CallInst>(&I);
  if (not Call)
    return false;

  // We have to hardcode revng_call_stack_arguments and revng_stack_frame
  // because SegregateStackAccesses has to mark them as functions that read
  // inaccessible memory, in order to prevent some LLVM optimizations.
  if (llvm::Function *Callee = getCalledFunction(Call)) {
    llvm::StringRef Name = Callee->getName();
    if (Name.startswith("revng_call_stack_arguments")
        or Name.startswith("revng_stack_frame"))
      return false;
  }

  // In all the other cases we can just use memory effects.
  auto CallMemoryEffects = Call->getMemoryEffects();
  bool MayAccessMemory = not CallMemoryEffects.doesNotAccessMemory();
  bool OnlyReadsMemory = CallMemoryEffects.onlyReadsMemory();
  return MayAccessMemory and OnlyReadsMemory;
}

/// Check if \a ModelType can be assigned to an llvm::Value of type \a LLVMType
/// during a memory operations (load, store and the like).
inline bool areMemOpCompatible(const model::Type &ModelType,
                               const llvm::Type &LLVMType,
                               const model::Binary &Model) {

  // loads/stores from/to void pointers are not allowed
  if (ModelType.isVoidPrimitive() or ModelType.isPrototype())
    return false;

  // We don't load or store entire structs in a single mem operation
  if (not ModelType.isScalar())
    return false;

  uint64_t ModelSize = ModelType.size().value();

  // For LLVM pointers, we want to check that the model type has the correct
  // size with respect to the current architecture
  if (LLVMType.isPointerTy()) {
    auto Size = model::Architecture::getPointerSize(Model.Architecture());
    return Size == ModelSize;
  }

  auto LLVMSize = LLVMType.getScalarSizeInBits();

  // Special case for i1
  if (LLVMSize < 8)
    return ModelSize == 1;

  return ModelSize * 8 == LLVMSize;
}

inline bool isAssignment(const llvm::Value *I) {
  return isCallToTagged(I, FunctionTags::Assign);
}

inline bool isComment(const llvm::Value *I) {
  return isCallToTagged(I, FunctionTags::Comment);
}

inline bool isLocalVarDecl(const llvm::Value *I) {
  return isCallToTagged(I, FunctionTags::LocalVariable);
}

inline bool isCallStackArgumentDecl(const llvm::Value *I) {
  auto *Call = dyn_cast_or_null<llvm::CallInst>(I);
  if (not Call)
    return false;

  auto *Callee = getCalledFunction(Call);
  if (not Callee)
    return false;

  return Callee->getName().startswith("revng_call_stack_arguments");
}

inline bool isArtificialAggregateLocalVarDecl(const llvm::Value *I) {
  return isCallToIsolatedFunction(I) and I->getType()->isAggregateType();
}

inline const llvm::CallInst *isCallToNonIsolated(const llvm::Value *I) {
  if (isCallToTagged(I, FunctionTags::QEMU)
      or isCallToTagged(I, FunctionTags::Helper)
      or isCallToTagged(I, FunctionTags::Exceptional)
      or llvm::isa<llvm::IntrinsicInst>(I))
    return llvm::cast<llvm::CallInst>(I);

  return nullptr;
}

inline bool isHelperAggregateLocalVarDecl(const llvm::Value *I) {
  return isCallToNonIsolated(I) and I->getType()->isAggregateType();
}

inline bool isStatement(const llvm::Instruction &I) {
  // Return are statements
  if (isa<llvm::ReturnInst>(I))
    return true;

  // Instructions that are not calls are never statement.
  auto *Call = dyn_cast<llvm::CallInst>(&I);
  if (not Call)
    return false;

  // Calls to Assign and LocalVariable are statemements.
  if (isAssignment(Call) or isComment(Call))
    return true;

  // Calls to isolated functions or helpers that return struct types on LLVM IR
  // need a statement.
  // This is necessary as a result of the fact that there is no direct mapping
  // between struct types on LLVM IR and on the model, so whenever a function
  // returns a struct in LLVM IR we cannot generally create a call to
  // LocalVariable nor to Copy/Assign (because we'd need to tag them with model
  // Type and we can't do that.), so we have to deal with it here on the fly.
  // We do it by marking these as statements, and emitting an assignment in C
  if (isArtificialAggregateLocalVarDecl(Call)
      or isHelperAggregateLocalVarDecl(Call))
    return true;

  // Calls to isolated functions and helpers that return void are statements.
  // If they don't return void, they are not statements. They are expressions
  // that will be assigned to some local variables in some other assign
  // statements.
  if (isCallToIsolatedFunction(Call) or isCallToNonIsolated(Call))
    return Call->getType()->isVoidTy();

  return false;
}
