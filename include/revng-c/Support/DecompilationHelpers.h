#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
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

    if (CalledFunc->isIntrinsic()) {
      bool IsReadOnly = CalledFunc->hasFnAttribute(llvm::Attribute::ReadOnly);
      bool IsReadNone = CalledFunc->hasFnAttribute(llvm::Attribute::ReadNone);
      return not IsReadOnly and not IsReadNone;
    }

    if (FunctionTags::Isolated.isTagOf(CalledFunc)
        or FunctionTags::WritesMemory.isTagOf(CalledFunc)
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

/// Decide whether a single instruction needs a top-scope variable or not.
inline bool needsTopScopeDeclaration(const llvm::Instruction &I) {
  const llvm::BasicBlock *CurBB = I.getParent();
  const llvm::BasicBlock &EntryBB = CurBB->getParent()->getEntryBlock();

  if (CurBB == &EntryBB) {
    // An instruction located in the first basic block of a function is already
    // in the top scope, so there is no need to add a separate variable for it.
    return false;
  }

  // If the instruction has uses outside its own basic block, we need a top
  // scope variable for it.
  // TODO: we can further refine this logic introducing the concept of
  // scopes and associating variable declarations to a scope.
  // For now, we decided to declare all variables that have at least one
  // use outside of their basic block right at the start of the
  // function, which is correct but overly conservative.
  auto HasDifferentParent = [Parent = I.getParent()](const llvm::User *U) {
    return cast<const llvm::Instruction>(U)->getParent() != Parent;
  };
  return any_of(I.users(), HasDifferentParent);
}
