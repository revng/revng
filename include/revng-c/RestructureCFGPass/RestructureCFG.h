#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/Pass.h"

#include "revng-c/RestructureCFGPass/ASTTree.h"
#include "revng-c/RestructureCFGPass/RegionCFGTreeBB.h"

class RestructureCFG : public llvm::FunctionPass {

private:
  using DuplicationMap = std::map<const llvm::BasicBlock *, size_t>;

public:
  static char ID;

public:
  RestructureCFG() : llvm::FunctionPass(ID), AST(), NDuplicates() {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  ASTTree &getAST() { return AST; }

  const DuplicationMap &getNDuplicates() const { return NDuplicates; }

private:
  ASTTree AST;
  DuplicationMap NDuplicates;
};
