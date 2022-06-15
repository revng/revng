#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"

namespace llvm {

class Function;
class Value;
class Instruction;

} // end namespace llvm

class ASTTree;

/// Check if the GHAST has loop dispatchers, which indicates the need for
/// a loop state variable to be declared.
bool hasLoopDispatchers(const ASTTree &GHAST);

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

/// Returns a set of all the llvm::Values for which we need a top-level
/// variable declaration.
llvm::SmallPtrSet<const llvm::Instruction *, 32>
collectTopScopeVariables(const llvm::Function &F);
