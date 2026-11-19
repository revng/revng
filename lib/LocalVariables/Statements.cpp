//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"

#include "revng/LocalVariables/LifetimeHelpers.h"
#include "revng/LocalVariables/Statements.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Support/Tag.h"

bool revng::isNotEmitted(const llvm::Instruction &I) {
  // The scope-graph markers only carry the structure of the scope graph.
  if (isCallToTagged(&I, FunctionTags::GotoBlockMarker)
      or isCallToTagged(&I, FunctionTags::ScopeCloserMarker)) {
    return true;
  }

  // A `llvm.lifetime.end` says the variable is dead from that point on, which
  // C expresses through the scope its declaration sits in and not through a
  // statement, so there is nothing to emit for it.
  //
  // Its sibling `llvm.lifetime.start` is deliberately not here: that one
  // carries the point where the variable is declared, and the clifter emits a
  // `clift.require` for it.
  return isLifetimeEnd(I);
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
