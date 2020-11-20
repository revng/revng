#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Pass.h"

namespace TypeShrinking {

class TypeShrinking : public llvm::FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  TypeShrinking() : FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F);
};

} // namespace TypeShrinking
