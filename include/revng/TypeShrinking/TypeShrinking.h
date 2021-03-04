#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/Scalar.h"

namespace TypeShrinking {

class TypeShrinking : public llvm::FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  TypeShrinking() : FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};

void applyTypeShrinking(llvm::legacy::FunctionPassManager &PM);

} // namespace TypeShrinking
