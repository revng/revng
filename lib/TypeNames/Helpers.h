#pragma once
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Casting.h"

#include "revng/Model/TypeDefinition.h"

inline bool
transitivelyRequiresCompleteSubtypeDefinition(const model::TypeDefinition &TD) {
  return TD.isStruct() or TD.isUnion();
}

inline bool requiresCompleteSubtypeDefinition(const model::TypeDefinition &TD) {
  return llvm::isa<model::StructDefinition>(TD)
         or llvm::isa<model::UnionDefinition>(TD);
}
