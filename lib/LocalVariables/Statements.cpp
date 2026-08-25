//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"

#include "revng/LocalVariables/Statements.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Support/Tag.h"

bool revng::isNotEmitted(const llvm::Instruction &I) {
  return isCallToTagged(&I, FunctionTags::GotoBlockMarker)
         or isCallToTagged(&I, FunctionTags::ScopeCloserMarker);
}

bool revng::isStatement(const llvm::Instruction &I) {
  if (revng::isNotEmitted(I))
    return false;

  // Terminators become `return`, `if`, `switch` or `goto` statements.
  if (I.isTerminator())
    return true;

  // Allocas become local variable declarations.
  if (llvm::isa<llvm::AllocaInst>(&I))
    return true;

  // Anything else is the root of an expression tree if nothing uses it, and
  // inlined into its users otherwise.
  return I.use_empty();
}

bool revng::isExpressionLeaf(const llvm::Value &V) {
  if (const auto *I = llvm::dyn_cast<llvm::Instruction>(&V))
    return isStatement(*I);

  // Arguments, constants, globals and block labels are one node each.
  return true;
}
