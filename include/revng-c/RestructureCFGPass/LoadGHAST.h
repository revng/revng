#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>

#include "llvm/Pass.h"

#include "revng-c/RestructureCFGPass/ASTTree.h"

class LoadGHASTWrapperPass : public llvm::ImmutablePass {

public:
  static char ID;

private:
  std::map<const llvm::Function *, ASTTree> AST;

public:
  LoadGHASTWrapperPass() : llvm::ImmutablePass(ID), AST() {}

public:
  bool doInitialization(llvm::Module &M) override final { return false; }

  bool doFinalization(llvm::Module &M) override final { return false; }

public:
  ASTTree &getGHAST(const llvm::Function &F) { return AST[&F]; }
};
