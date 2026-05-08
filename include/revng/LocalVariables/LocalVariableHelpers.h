#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"

/// \name Functions for manipulating stack model type metadata
///
///@{

inline constexpr const char *StackFrameMDName = "revng.stack_frame";

inline bool hasStackFrameMetadata(const llvm::AllocaInst *I) {
  return I->hasMetadata(I->getContext().getMDKindID(StackFrameMDName));
};

inline void setStackFrameMetadata(llvm::AllocaInst *A) {
  A->setMetadata(StackFrameMDName, llvm::MDNode::get(A->getContext(), {}));
}

///@}
