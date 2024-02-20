#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Type.h"

#include "revng/Model/Architecture.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Type.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/Support/FunctionTags.h"

/// Returns true if I should be considered with side effects for decompilation
/// purposes.
inline bool hasSideEffects(const llvm::Instruction &I) {
  if (isa<llvm::StoreInst>(&I))
    return true;

  if (auto *Call = llvm::dyn_cast<llvm::CallInst>(&I)) {
    auto *CalledFunc = Call->getCalledFunction();
    if (not CalledFunc)
      return true;

    if (isCallToIsolatedFunction(Call))
      return true;

    if (CalledFunc->isIntrinsic()) {
      bool IsReadOnly = CalledFunc->onlyReadsMemory();
      bool IsReadNone = CalledFunc->doesNotAccessMemory();
      return not IsReadOnly and not IsReadNone;
    }

    if (FunctionTags::WritesMemory.isTagOf(CalledFunc)
        or FunctionTags::Helper.isTagOf(CalledFunc)
        or FunctionTags::QEMU.isTagOf(CalledFunc)
        or FunctionTags::Exceptional.isTagOf(CalledFunc))
      return true;
  }

  return false;
}

/// Check if \a ModelType can be assigned to an llvm::Value of type \a LLVMType
/// during a memory operations (load, store and the like).
inline bool areMemOpCompatible(const model::QualifiedType &ModelType,
                               const llvm::Type &LLVMType,
                               const model::Binary &Model) {

  // We don't load or store entire structs in a single mem operation
  if (not ModelType.isScalar())
    return false;

  // loads/stores from/to void pointers are not allowed
  if (ModelType.isVoid())
    return false;

  auto ModelSize = ModelType.size().value();

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

inline bool isLocalVarDecl(const llvm::Value *I) {
  return isCallToTagged(I, FunctionTags::LocalVariable);
}

inline bool isCallStackArgumentDecl(const llvm::Value *I) {
  auto *Call = dyn_cast_or_null<llvm::CallInst>(I);
  if (not Call)
    return false;

  auto *Callee = Call->getCalledFunction();
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
