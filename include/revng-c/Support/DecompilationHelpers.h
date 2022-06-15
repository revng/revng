#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Type.h"

#include "revng/Model/Architecture.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Type.h"

#include "revng-c/Support/FunctionTags.h"

/// Returns true if F is an llvm::Function should be considered pure for
/// decompilation purposes
inline bool isPure(const llvm::Function *F) {
  if (F) {
    if (FunctionTags::ModelGEP.isTagOf(F)
        or FunctionTags::StructInitializer.isTagOf(F)
        or FunctionTags::AddressOf.isTagOf(F)
        or FunctionTags::OpaqueCSVValue.isTagOf(F)) {
      return true;
    }
  }
  return false;
}

/// Returns true if I is a call to an llvm::Function that should be considered
/// pure for decompilation purposes
inline bool isCallToPure(const llvm::Instruction &I) {
  if (auto *Call = dyn_cast<llvm::CallInst>(&I))
    return isPure(Call->getCalledFunction());

  return false;
}

/// Check if \a ModelType can be assigned to an llvm::Value of type \a LLVMType
/// during a memory operations (load, store and the like).
inline bool areMemOpCompatible(const model::QualifiedType &ModelType,
                               const llvm::Type &LLVMType,
                               const model::Binary &Model) {

  // We don't load or store entire structs in a single mem operation
  if (not ModelType.isPointer() and not ModelType.isScalar())
    return false;

  auto ModelSize = ModelType.size().value();

  // For LLVM pointers, we want to check that the model type has the correct
  // size with respect to the current architecture
  if (LLVMType.isPointerTy()) {
    auto PointerSize = model::Architecture::getPointerSize(Model.Architecture);
    return PointerSize == ModelSize;
  }

  auto LLVMSize = LLVMType.getScalarSizeInBits();

  // Special case for i1
  if (LLVMSize < 8)
    return ModelSize == 1;

  return ModelSize * 8 == LLVMSize;
}
