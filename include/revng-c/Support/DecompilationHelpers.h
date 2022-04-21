#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"

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
