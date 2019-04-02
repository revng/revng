#ifndef REVNGC_RESTRUCTURE_CFG_RESTRUCTURECFG_H
#define REVNGC_RESTRUCTURE_CFG_RESTRUCTURECFG_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <fstream>

// LLVM includes
#include "llvm/Pass.h"

// Local libraries includes
#include "revng-c/RestructureCFGPass/RegionCFGTree.h"

// Local includes
#include "ReachabilityPass.h"

// Forward reference to object types
class ASTTree;

class RestructureCFG : public llvm::FunctionPass {

private:
  using DuplicationMap = std::map<llvm::BasicBlock *, size_t>;

protected:
  bool Done;

public:
  static char ID;

public:
  RestructureCFG() : llvm::FunctionPass(ID), Done(false) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  ASTTree &getAST() { return RootCFG.getAST(); }

  RegionCFG &getRCT() { return RootCFG; }

  bool isDone() { return Done; }

  std::map<llvm::BasicBlock*, size_t> &getNDuplicates() { return NDuplicates; }

private:
  RegionCFG RootCFG;
  DuplicationMap NDuplicates;
};

#endif // REVNGC_RESTRUCTURE_CFG_RESTRUCTURECFG_H
