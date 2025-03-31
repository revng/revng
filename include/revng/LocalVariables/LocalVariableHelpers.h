#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"

inline constexpr const char *StackTypeMDName = "revng.stack_type";
inline const char *VariableTypeMDName = "revng.variable_type";

inline bool hasNamedMetadata(const llvm::AllocaInst *I,
                             llvm::StringRef MDName) {
  return nullptr != I->getMetadata(I->getContext().getMDKindID(MDName));
}

/// \name Functions for manipulating stack model type metadata
///
///@{

inline bool hasStackTypeMetadata(const llvm::AllocaInst *I) {
  return hasNamedMetadata(I, StackTypeMDName);
};

void setStackTypeMetadata(llvm::AllocaInst *I, const model::Type &StackType);

model::UpcastableType getStackTypeFromMetadata(const llvm::AllocaInst *I,
                                               const model::Binary &Model);

///@}

/// \name Helper functions for manipulating local variable model type metadata
///
///@{

inline bool hasVariableTypeMetadata(const llvm::AllocaInst *I) {
  return hasNamedMetadata(I, VariableTypeMDName);
}

void setVariableTypeMetadata(llvm::AllocaInst *I,
                             const model::Type &VariableType);

model::UpcastableType getVariableTypeFromMetadata(const llvm::AllocaInst *I,
                                                  const model::Binary &Model);

///@}
